import asyncio
import json
import re

from model import llm
from firebase_service import FirebaseService
from pydantic import BaseModel
from general import app

class DecisionRequest(BaseModel):
    requirement: str

class DecisionResponse(BaseModel):
    decision: str


def extract_json(text):
    text = text.strip()

    # remove markdown
    text = text.replace("```json", "")
    text = text.replace("```", "")

    start = text.find("{")
    end = text.rfind("}") + 1

    if start == -1 or end == 0:
        raise ValueError("No JSON found")

    json_str = text[start:end]

    try:
        return json.loads(json_str)

    except json.JSONDecodeError as e:
        print("\n========== INVALID JSON ==========")
        print(json_str)
        print("==================================\n")
        raise e

class DecisionMakingAgent:
    def __init__(self, llm):
        self.llm = llm

    def get_relevant_product(self, requirements):
        """
        🌐 Line-by-Line Firebase Cloud Vector RAG.
        Generates individual embedding vectors per requirement line to prevent
        CCTV camera crowding, using pure vector queries to completely avoid
        composite index precondition errors.
        """

        DISTANCE_THRESHOLD = 0.50
        retrieval_catalog = []
        CATEGORY_LIMITS = {
            "CCTV Camera": 20,
            "Recorder": 5,
            "Hard Drive": 5,
            "Mounts & Enclosures": 5,
            "Switches": 5,
            "Cable": 5
        }

        for category, limit in CATEGORY_LIMITS.items():
            matched_docs = FirebaseService().find_nearest_catalog(requirements, category=category, limit=limit)

            for doc in matched_docs:
                data = doc.to_dict()
                distance = data.get("vector_distance", 1.0)
                item_category = data.get("category", "Unknown")

                if distance <= DISTANCE_THRESHOLD:
                    doc_id_clean = str(doc.id).strip()

                    if not any(x['ID'] == doc_id_clean for x in retrieval_catalog):
                        retrieval_catalog.append({
                            "ID": doc_id_clean,
                            "Product Name": data.get("product_name", "Unknown Component"),
                            "Features": data.get("features", []),
                            "Price": float(data.get("price", 0.0)),
                            "Description": data.get("description", ""),
                            "Category": item_category
                        })

        retrieval_catalog.sort(key=lambda x: x.get("vector_distance", 1.0))

        print(f"\n[Success] Cloud RAG compiled. Pooled {len(retrieval_catalog)} targeted components safely.")

        print(f"\n📋 [RAG Retrieval Catalog Sample Output]: {retrieval_catalog}")

        context_string = "\n---\n".join(str(item) for item in retrieval_catalog)
        return {
            "retrieval_catalog_string": context_string,
            "retrieval_catalog_json": retrieval_catalog
        }

    def optimize_plan(self, requirements, retrieval_catalog_str, previous_feedback=""):
        """Agent 1: Proposes a complete Turnkey System plan using Alphanumeric IDs and Quantities."""
        prompt = f"""
        Act as an expert Enterprise Security Solutions Architect and System Integration Engineer.
        Your critical goal is to select the optimal combination of equipment from the Retrieved Catalog to design a fully operational, complete, and integrated turnkey Security System solution.

        [MANDATORY SURVEILLANCE INFRASTRUCTURE SYSTEM LAWS]
        A functioning surveillance network cannot exist on individual components alone. You MUST analyze the user requirements and include components across EVERY single category requested:
        1. FRONT-END CAMERAS: Select appropriate camera models and amounts to fully satisfy the requested camera counts.
        2. BACKEND RECORDER (DVR/NVR): You MUST explicitly add an appropriate central recording server (NVR) from the catalog. NEVER leave this category missing or deleted under any circumstances.
        3. DATA STORAGE DRIVE: Recorders require internal media storage. You MUST add constant 24/7 duty cycle surveillance hard drives from the catalog. Optimize the drive size (e.g., choose 8TB or higher if 4K recording metrics require heavy capacity thresholds).
        4. PoE SWITCHES (NETWORKING): Select an appropriate networking switch from the catalog to power the deployment network layout based on total camera port loads.
        5. CABLES EXPERT SELECTION RULE: If the user requirement for 'Cable' contains no features or specifications, automatically select an appropriate high-quality Cat6 UTP network cable box (e.g., PFM920I-6UN-C) to support PoE power and digital stream transmission.

        User Requirements: {json.dumps(requirements, indent=2)}
        Available Catalog (Valid Options Map): {retrieval_catalog_str}
        Previous Review Feedback (if any, please FIX these issues but NEVER delete required category items): {previous_feedback}

        MODE:
        1. Disable think mode. You MUST NOT use any <think> or similar tags in your response. Focus on delivering a clean, structured JSON output ONLY.

        RULES:
        1. You must select products ONLY from the Retrieved Catalog. Match them exactly using their string ID values (Alphanumeric).
        2. Stay strictly within the targeted budget limits.
        3. You MUST propose a complete system solution that includes components across ALL requested categories. NEVER omit an entire category of equipment. If a category is requested, it MUST be represented in the proposed plan with at least one product ID and quantity. For example, if 'Cameras' are requested, you cannot propose a plan that only includes NVRs and Switches without any camera IDs.
        4. You MUST make sure the quantities of each selected product perfectly match the user requirements. If the user requests 5 cameras, you MUST propose a quantity of 5 for the selected camera model. Do NOT under-propose or over-propose quantities.
        5. The total number of proposed quantity of cameras MUST match the total quantity requested in the user requirements. For example, if the user requests a total of 10 items of cameras, the sum of the quantities of all cameras MUST equal 10. Do NOT propose a total quantity that is less than or greater than what is requested. You MUST use calculations code to calculate the total quantity, instead of relying on the LLM to do it correctly in its head.
        5. Every category requested in User Requirements MUST have at least one product ID mapped. Never drop a category entirely.

        STRICT FORMAT RULES:
        - Output ONLY valid JSON structure matching the blueprint schema model below.
        - A JSON structure MUST use "," as a separator and MUST NOT use any bullet points, lists, or markdown formatting.
        - A JSON structure MUST use double quotes for all keys and string values. Single quotes are NOT allowed.
        - No explanation, no conversational text, no markdown labels or ticks.

        Return ONLY a JSON array structure following this exact format:
        {{
            "proposed_items": [
                {{"id": "SELECTED_PRODUCT_ID_1", "quantity": 5}},
                {{"id": "SELECTED_PRODUCT_ID_2", "quantity": 1}},
                {{"id": "SELECTED_PRODUCT_ID_3", "quantity": 2}}
            ]
        }}
        """

        output = self.llm.generate(prompt, temperature=0, json_mode=True)
        res = extract_json(output)
        return res.get("proposed_items", [])

    def extract_product_detail(self, proposed_items, retrieval_catalog):
        """Extracts actual prices and features from the RAG data to prevent hallucinations."""
        catalog_lookup = {str(item["ID"]).strip(): item for item in retrieval_catalog}
        plan = {"selected_items": [], "total_cost": 0.0}

        for item in proposed_items:
            prod_id = str(item.get("id", item.get("ID", ""))).strip()
            qty = int(item.get("quantity", 0))

            if prod_id in catalog_lookup and qty > 0:
                real_product = catalog_lookup[prod_id]
                unit_price = float(real_product.get("Price", 0.0))
                subtotal = unit_price * qty

                plan["selected_items"].append({
                    "id": prod_id,
                    "name": real_product.get("Product Name", "Unknown"),
                    "features": real_product.get("Features", []),
                    "category": real_product.get("Category", "Hardware"),
                    "description": real_product.get("Description", ""),
                    "quantity": qty,
                    "unit_price": unit_price,
                    "subtotal": subtotal
                })
                plan["total_cost"] += subtotal

        plan["total_cost"] = round(plan["total_cost"], 2)
        return plan

    def review_plan(self, plan, requirements):
        """Agent 2: Reviews the strictly assembled plan for full turnkey system completeness and budget constraints."""
        prompt = f"""
        Act as the System Constraint Review Agent.
        Review the proposed integrated security configuration plan against the original requirements and technical design rules.

        User Requirements: {json.dumps(requirements, indent=2)}
        Proposed Plan: {json.dumps(plan, indent=2)}

        MODE:
        1. Disable think mode. You MUST NOT use any <think> or similar tags in your response. Focus on delivering a clean, structured JSON output ONLY.

        CRITICAL EVALUATION AUDITS (PRIORITIZE SPECS & FEATURES OVER BUDGET MAXIMIZATION):
        1. System Completeness & Quantities: Verify if the plan spans across a complete turnkey operational infrastructure. It MUST include front-end Surveillance Cameras, a Central Recorder (NVR), data storage Hard Drives, network PoE Switches, and Cable accessories. Ensure the requested item quantities match perfectly. Deduct score heavily (score < 40) if any core category is missing.
        2. Technical Specifications & Feature Fit (Highest Priority): Audit how accurately the chosen item specifications fulfill the explicit feature requirements requested by the user (e.g., matching 8MP/4K resolutions, specific housing styles like dome/bullet, weatherproofing parameters, and night vision IR ranges).
        3. Financial Compliance (NO PENALTY FOR BEING UNDER BUDGET): Verify if the Total Cost is within the Total Target Budget parameters.
           *CRITICAL RULE*: Being significantly lower than the total target budget is considered an exceptional technical benefit and highly cost-effective. Do NOT deduct points, do NOT flag it as 'underutilization', and do NOT lower the score for a lower overall system cost, provided that all quantity, structural, and technical feature requests are successfully matched.

        Return ONLY JSON:
        {{
            "score": 95,
            "feedback": "Explain step-by-step what is technically valid, and what components are missing or require changes based on technical compliance."
        }}
        """
        print("[Review Agent] Auditing system composition and topology boundaries...")
        output = self.llm.generate(prompt, temperature=0, json_mode=True)
        res = extract_json(output)
        return {
            "score": res.get("score", 0),
            "feedback": res.get("feedback", "No feedback provided.")
        }

    def reasoning_plan(self, requirements, plan):
        print("--> [Reasoner] Drafting final proposal report...")
        prompt = f"""
        Act as an expert Enterprise Security Solutions Technical Sales Engineer.
        Write a professional, comprehensive B2B deployment proposal based on the successful plan in a message format (Strictly no email markers and no json format).

        CRITICAL RECOGNITION: Use clean terminology like "Surveillance Systems", "Network Video Recording Channels", or "IP Camera Arrays" throughout the proposal prose text.

        Proposed Turnkey Bill of Materials (BOM): {json.dumps(plan['selected_items'], indent=2)}.
        Total Combined System Cost: ${plan['total_cost']}.
        Proposed Plan Audit Logs: {json.dumps(plan["review_feedback"], indent=2)}.
        User System Requirements Specification: {json.dumps(requirements, indent=2)}.

        Your proposal layout must explicitly deliver:
        1. Executive Summary: Confirming overall project budget compliance and outstanding cost efficiency.
        2. Front-End Surveillance Layout: Highlighting the choice of designated cameras.
        3. Back-End Core Networking & Storage Dependency Analysis: Explaining why incorporating the central NVR recorder server, specific data transmission PoE network switches, and surveillance-grade constant read/write Hard Disk storage drives is structurally mandatory to achieve system functionality.
        4. Cabling Infrastructure & Deployment Ready scalability parameters.

        MODE:
        1. Disable think mode. You MUST NOT use any <think> or similar tags in your response. Focus on delivering a clean, structured JSON output ONLY.

        RULES:
        Your reply is sent DIRECTLY to the client. Never output internal instructions, decision logic, bullet-point rules, or conditional statements
        """
        return self.llm.generate(prompt, temperature=0.1)

    def run(self, requirements, retrieval_catalog):
        """Main Loop Engine executing the workflow."""
        MAX_ITERATIONS = 3
        SCORE_THRESHOLD = 70

        best_plan = None
        highest_score = 0
        current_feedback = ""

        for attempt in range(MAX_ITERATIONS):
            print(f"\n--- Attempt {attempt + 1} of {MAX_ITERATIONS} ---")

            proposed_items = self.optimize_plan(requirements, retrieval_catalog["retrieval_catalog_string"], current_feedback)
            plan = self.extract_product_detail(proposed_items, retrieval_catalog["retrieval_catalog_json"])

            print(f"📋 [Attempt {attempt + 1} Detailed Bill of Materials Output]:")
            if not plan["selected_items"]:
                print("   ❌ No items successfully selected or matched.")
            for idx, item in enumerate(plan["selected_items"], 1):
                print(f"   {idx}. [{item['category']}] ID: {item['id']} | {item['name']} | Qty: {item['quantity']} | Unit: ${item['unit_price']} | Subtotal: ${item['subtotal']}")
            print(f"   💰 Proposed Current Subtotal: ${plan['total_cost']}")

            review = self.review_plan(plan, requirements)
            score = review["score"]

            print(f"Review Score: {score}/100")
            print(f"Review Feedback: {review['feedback']}\n")

            if score > highest_score:
                highest_score = score
                best_plan = plan
                best_plan["review_score"] = score
                best_plan["review_feedback"] = review["feedback"]

            if score >= SCORE_THRESHOLD:
                print("Target system completeness optimization reached. Finalizing plan.")
                break
            else:
                current_feedback = review["feedback"]

        print(f"\n--- Process Finished. Best Infrastructure Score Achieved: {highest_score}/100 ---")
        reasoning = self.reasoning_plan(requirements, best_plan)
        return {
            "final_data_structure": best_plan,
            "b2b_proposal_report": reasoning
        }
    
@app.post("/decision", response_model=DecisionResponse)
def generate_decision_endpoint(request: DecisionRequest) -> DecisionResponse:
    print("📋 [Requirement Parser] Starting parsing of primitive requirements across all hardware categories...")

    # Initialize agent proxy controllers
    decisionMaker = DecisionMakingAgent(llm)

    # Execute Cloud RAG Vector Search
    retrieval_catalog = decisionMaker.get_relevant_product(request.requirement)

    # Drive Multi-Agent Iteration Streams
    output_result = decisionMaker.run(request.requirement, retrieval_catalog)

    # Unpack and map final structures onto console interfaces
    final_plan = output_result["final_data_structure"]

    print("\n" + "="*60)
    print("🏆 FINAL CHOSEN TURNKEY SOLUTION PLAN SUMMARY")
    print("="*60)
    print(f"📊 Best Engineering Fit Score: {final_plan['review_score']}/100")
    print(f"💵 Cumulative System Contract Price: ${final_plan['total_cost']}")
    print("-"*60)
    print("📦 SELECTED BILL OF MATERIALS (BOM):")

    for idx, item in enumerate(final_plan["selected_items"], 1):
        print(f"  [{idx}] Category: {item['category']}")
        print(f"      Model ID: {item['id']}")
        print(f"      Product Name: {item['name']}")
        print(f"      Quantity: {item['quantity']} units")
        print(f"      Financial Metric: ${item['unit_price']} each | Subtotal: ${item['subtotal']}")
        print(f"      Specs: {item['description']}\n")

    print("-"*60)
    print("📝 TECHNICAL SALES ENG B2B DEPLOYMENT PROPOSAL REPORT:")
    print("-"*60)
    print(output_result["b2b_proposal_report"])

    print("\n=======================================================")
    print("[SYSTEM EXECUTION CYCLE CONCLUDED SUCCESSFULLY]")
    print("=======================================================")

    return DecisionResponse(decision=output_result["b2b_proposal_report"])