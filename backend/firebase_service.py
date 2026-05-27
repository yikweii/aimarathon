"""Firebase service for CCTV Requirements Chatbot."""

import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional

import firebase_admin
from firebase_admin import credentials, firestore, storage
from dotenv import load_dotenv

from google.cloud.firestore_v1.base_vector_query import DistanceMeasure
from google.cloud.firestore_v1.vector import Vector
from google.cloud.firestore_v1 import FieldFilter

from model import llm

load_dotenv()


def _now() -> datetime:
    return datetime.utcnow()


class FirebaseService:
    """Singleton — wraps Firestore, Storage, and Auth for the CCTV chatbot."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_firebase()
        return cls._instance

    def _init_firebase(self):
        key_path = os.getenv("FIREBASE_KEY_PATH", "./firebase-key.json")
        if not os.path.exists(key_path):
            raise FileNotFoundError(
                f"Firebase key not found at {key_path}. "
                "Download from Firebase Console → Project Settings → Service Accounts."
            )
        try:
            firebase_admin.initialize_app(
                credentials.Certificate(key_path),
                {"storageBucket": os.getenv("FIREBASE_STORAGE_BUCKET")},
            )
        except ValueError:
            pass  # Already initialised

        self.db = firestore.client()
        self.bucket = storage.bucket()
        print("✓ Firebase initialised")

    # ── helpers ──────────────────────────────────────────────────────────────

    def _col(self, name: str):
        return self.db.collection(name)

    def _doc(self, col: str, doc_id: str):
        return self._col(col).document(doc_id)

    def _new_id(self) -> str:
        return str(uuid.uuid4())

    # ── Projects ─────────────────────────────────────────────────────────────

    def create_project(self, data: Dict) -> str:
        pid = self._new_id()
        self._doc("projects", pid).set({**data, "created_at": _now(), "updated_at": _now()})
        print(f"✓ Project created: {pid}")
        return pid

    def get_project(self, pid: str) -> Optional[Dict]:
        doc = self._doc("projects", pid).get()
        return {**doc.to_dict(), "project_id": pid} if doc.exists else None

    def update_project(self, pid: str, updates: Dict) -> bool:
        try:
            self._doc("projects", pid).update({**updates, "updated_at": _now()})
            return True
        except Exception as e:
            print(f"✗ update_project: {e}")
            return False

    def list_projects(self, limit: int = 50) -> List[Dict]:
        docs = (
            self._col("projects")
            .order_by("created_at", direction=firestore.Query.DESCENDING)
            .limit(limit)
            .stream()
        )
        return [{**d.to_dict(), "project_id": d.id} for d in docs]

    def delete_project(self, pid: str) -> bool:
        try:
            batch = self.db.batch()
            for conv in self._col("conversations").where("project_id", "==", pid).stream():
                for msg in conv.reference.collection("messages").stream():
                    batch.delete(msg.reference)
                batch.delete(conv.reference)
            batch.delete(self._doc("requirements", pid))
            batch.delete(self._doc("projects", pid))
            batch.commit()
            print(f"✓ Project deleted: {pid}")
            return True
        except Exception as e:
            print(f"✗ delete_project: {e}")
            return False

    # ── Conversations & Messages ──────────────────────────────────────────────

    def create_conversation(self, pid: str) -> str:
        cid = self._new_id()
        self._doc("conversations", cid).set(
            {"project_id": pid, "created_at": _now(), "updated_at": _now(), "message_count": 0}
        )
        return cid

    def save_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        floor_plan_url: Optional[str] = None,
    ) -> str:
        mid = self._new_id()
        msg = {"role": role, "content": content, "timestamp": _now()}
        if floor_plan_url:
            msg["floor_plan_url"] = floor_plan_url

        self._doc("conversations", conversation_id).collection("messages").document(mid).set(msg)
        self._doc("conversations", conversation_id).update(
            {"updated_at": _now(), "message_count": firestore.Increment(1)}
        )
        return mid

    def batch_save_conversation(self, pid: str, messages: List[Dict]) -> str:
        cid = self.create_conversation(pid)
        batch = self.db.batch()
        for msg in messages:
            mid = self._new_id()
            data = {"role": msg.get("role", "user"), "content": msg.get("content", ""), "timestamp": _now()}
            if msg.get("floor_plan_url"):
                data["floor_plan_url"] = msg["floor_plan_url"]
            batch.set(self._doc("conversations", cid).collection("messages").document(mid), data)
        batch.commit()
        print(f"✓ Batch: {len(messages)} messages → {cid}")
        return cid

    def get_conversation_history(self, cid: str) -> List[Dict]:
        docs = (
            self._doc("conversations", cid)
            .collection("messages")
            .order_by("timestamp")
            .stream()
        )
        result = []
        for d in docs:
            msg = {**d.to_dict(), "message_id": d.id}
            if isinstance(msg.get("timestamp"), datetime):
                msg["timestamp"] = msg["timestamp"].isoformat()
            result.append(msg)
        return result

    def get_project_history(self, pid: str) -> Dict:
        convs = []
        for d in self._col("conversations").where("project_id", "==", pid).stream():
            conv = {**d.to_dict(), "conversation_id": d.id, "messages": self.get_conversation_history(d.id)}
            convs.append(conv)
        return {"project_id": pid, "conversation_count": len(convs), "conversations": convs}

    # ── Requirements ─────────────────────────────────────────────────────────

    def save_requirements(
        self,
        project_id: str,
        requirements: Dict,
        stage: str,
        summary: Optional[str] = None,
        optimization_summary: Optional[str] = None,
        progress: int = 0,
        inferred_fields: Optional[List[str]] = None,
    ) -> bool:
        """
        Incrementally update requirements — never overwrites fields from a
        previous richer extraction. Uses Firestore dot-notation so only the
        fields present in this extraction pass are touched.

        Document shape  (requirements/{project_id}):
            stage, progress, updated_at, summary, inferred_fields
            data : {
                site_type, location, size, budget, budget_tier,
                coverage_areas[], coverage_focus, priority_zone,
                resolution, night_vision, remote_viewing, retention_days,
                installation, connectivity, timeline,
                has_floor_plan, layout_type, identified_zones[],
                entry_points[], blind_spots[],
                features[], keywords[]
            }
        """
        try:
            doc_ref = self._doc("requirements", project_id)

            # Build dot-notation update so each field merges into data{}
            # without touching fields not present in this extraction pass.
            update: Dict = {
                "stage": stage,
                "progress": progress,
                "updated_at": _now(),
                "inferred_fields": inferred_fields or [],
            }
            if summary:
                update["summary"] = summary
            if optimization_summary:
                update["optimization_summary"] = optimization_summary

            for key, value in requirements.items():
                # Skip empty / null values — don't overwrite a real value with nothing
                if value is None or value == "" or value == [] or value == {}:
                    continue
                update[f"data.{key}"] = value

            # update() merges into existing doc; falls back to set() on first write
            try:
                doc_ref.update(update)
            except Exception:
                # Document doesn't exist yet
                doc_ref.set({
                    "stage": stage,
                    "progress": progress,
                    "updated_at": _now(),
                    "summary": summary or "",
                    "inferred_fields": inferred_fields or [],
                    "data": requirements,
                })

            print(f"✓ Requirements [{progress}%] stage={stage}: {project_id}")
            return True
        except Exception as e:
            print(f"✗ save_requirements: {e}")
            return False

    def get_requirements(self, project_id: str) -> Optional[Dict]:
        doc = self._doc("requirements", project_id).get()
        return {**doc.to_dict(), "project_id": project_id} if doc.exists else None

    # ── Cloud Storage ─────────────────────────────────────────────────────────

    def upload_floor_plan(self, file_path: str, pid: str) -> Optional[str]:
        try:
            path = f"floor_plans/{pid}/{os.path.basename(file_path)}"
            blob = self.bucket.blob(path)
            blob.upload_from_filename(file_path)
            blob.make_public()
            print(f"✓ Floor plan uploaded: {path}")
            return blob.public_url
        except Exception as e:
            print(f"✗ upload_floor_plan: {e}")
            return None

    def delete_floor_plan(self, storage_path: str) -> bool:
        try:
            self.bucket.blob(storage_path).delete()
            return True
        except Exception as e:
            print(f"✗ delete_floor_plan: {e}")
            return False

    # ── Health ────────────────────────────────────────────────────────────────

    def health_check(self) -> Dict:
        try:
            self._col("projects").limit(1).stream()
            return {"status": "healthy", "firebase": "connected"}
        except Exception as e:
            return {"status": "unhealthy", "firebase": f"error: {e}"}
        
    def find_nearest_catalog(self, text, category=None, limit=5):

        line_vector = llm.embedding(text)

        collection_ref = self.db.collection("catalog")

        # build base query
        query_ref = collection_ref

        if category:
            query_ref = collection_ref.where(
                filter=FieldFilter("category", "==", category)
        )

        return query_ref.find_nearest(
            vector_field="embedding_vector",
            query_vector=Vector(line_vector),
            distance_measure=DistanceMeasure.COSINE,
            limit=limit,
            distance_result_field="vector_distance"
        ).stream()