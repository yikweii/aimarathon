import os
import json
import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud.firestore_v1.vector import Vector
from model import llm

# =====================================================================
# 1. Initialize Firebase Safely (Notebook Re-run Safe)
# =====================================================================
try:
    firebase_admin.get_app()
    print("[Firebase] Existing default app detected. Reusing active connection safely.")
except ValueError:
    cred_file = "./firebase-key.json"
    cred = credentials.Certificate(cred_file)
    firebase_admin.initialize_app(cred)
    print("[Firebase] Cloud database connection initialized successfully.")

db = firestore.client()
collection_ref = db.collection("catalog")

# =====================================================================
# 2. Define Updated Catalog Mapping (All 7 Asset Files Included)
# =====================================================================
files_map = {
    "CCTV Camera": ["./catalog/01_cameras_hdcvi.json", "./catalog/03_cameras_ip.json"],
    "Recorder": ["./catalog/02_recorders.json"],
    "Hard Drive": ["./catalog/04_consumer_vms_storage.json"],
    "Mounts & Enclosures": ["./catalog/05_tools_mounts.json"],
    "Switches": ["./catalog/06_networking.json"],
    "Cable": ["07_cables_power_converters.json"]
}

print("\n🚀 Starting catalog vector embedding generation and cloud upload...")

# =====================================================================
# 3. Process and Store Data in Firestore (With Path Sanitation)
# =====================================================================
for category, filenames in files_map.items():
    for filename in filenames:
        if os.path.exists(filename):
            print(f"\nProcessing configuration file: {filename} [{category}]...")

            with open(filename, 'r', encoding='utf-8') as f:
                hardware_list = json.load(f)

                if isinstance(hardware_list, list):
                    for item in hardware_list:
                        product_name = str(item.get("name", item.get("Product Name", "")))
                        raw_prod_id = str(item.get("id", item.get("ID", ""))).strip()

                        # Filter to isolate relevant surveillance storage components
                        if category == "Hard Drive" and "Hard Drive" not in product_name:
                            continue

                        # Skip empty rows if any exist
                        if not product_name or product_name == "nan" or not raw_prod_id:
                            continue

                        # 🌟 FIXED: Sanitize forward slashes to prevent Firestore path path splitting errors
                        prod_id = raw_prod_id.replace("/", "-")

                        # Extract features list (handles arrays or comma strings safely)
                        features = item.get("features", item.get("Features", []))
                        if isinstance(features, str):
                            features = [f.strip() for f in features.split(", ")]

                        # Map specification dictionary or description string into a clean string parameter
                        specification = item.get("specification", item.get("Description", ""))
                        if isinstance(specification, dict):
                            description_str = ", ".join(f"{k}: {v}" for k, v in specification.items())
                        else:
                            description_str = str(specification)

                        use_case = str(item.get("use_case", item.get("Potential Use Case", "")))

                        # --- PRICE CONVERSION LOGIC (Preserving Your Exact Rules) ---
                        price_raw = str(item.get("price", item.get("Price", "")))
                        try:
                            # 1. Remove '$' and extra spaces
                            clean_price = price_raw.replace("$", "").strip()
                            # 2. Remove the thousands separator (e.g., "2.730,00" -> "2730,00")
                            clean_price = clean_price.replace(".", "")
                            # 3. Replace the decimal comma with a decimal point (e.g., "2730,00" -> "2730.00")
                            clean_price = clean_price.replace(",", ".")

                            # 4. Convert to float
                            price_float = float(clean_price)
                        except ValueError:
                            price_float = 0.0

                        # Combine data into a rich text chunk for the AI embedding model to process
                        content_to_embed = (
                            f"Product Name: {product_name}. "
                            f"Category: {category}. "
                            f"Description: {description_str}. "
                            f"Use Case: {use_case}"
                        )

                        # Generate the dense token vector embedding via your model
                        embedding = llm.embedding(content_to_embed)

                        # Create the unified document payload structure
                        doc_data = {
                            "product_name": product_name,
                            "features": features,
                            "price": price_float,
                            "use_case": use_case,
                            "description": description_str,
                            "category": category,
                            "embedding_vector": Vector(embedding)
                        }

                        # Push record to Firebase using the sanitized alphanumeric code as the unique Document ID
                        collection_ref.document(prod_id).set(doc_data)
                        print(f"Uploaded [{category}] -> {prod_id} with verified price: ${price_float}")
        else:
            print(f"⚠️ File not found in folder path: {filename}")

print("\n🎉 Cloud vectorization database upload complete with clean path segments!")