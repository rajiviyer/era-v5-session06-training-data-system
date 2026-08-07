#!/usr/bin/env python3
"""Regenerate data/toy_corpus/documents.jsonl with P1-T04 demo content."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

_ASSIGNMENT = Path(__file__).resolve().parents[1]
_SRC = _ASSIGNMENT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from corpus.loader import load_corpus  # noqa: E402
from corpus.schema import content_sha256, validate_document_record  # noqa: E402

LOCAL_PATH = "data/toy_corpus/documents.jsonl"
COMMON = {
    "content_status": "ready",
    "license_tier": "safe",
    "cleaning_pipeline_hash": "clean_toy_v1",
    "dedup_status": "passed",
    "pii_screen_status": "screened",
    "eval_overlap_status": "clear",
    "local_path": LOCAL_PATH,
}

TEXT: dict[str, str] = {
    "doc-web-001": (
        "Large language models learn statistical patterns over billions of tokens. "
        "Pretraining mixes web pages, books, and curated datasets so the model acquires "
        "grammar, facts, and reasoning priors before task-specific fine-tuning."
    ),
    "doc-web-002": (
        "Tokenization splits raw text into subword units. Byte Pair Encoding merges frequent "
        "character pairs until the vocabulary reaches a fixed size. The same tokenizer must "
        "stay frozen once pretraining begins so token IDs remain stable across shards."
    ),
    "doc-web-003": (
        "India's digital public infrastructure stack includes Aadhaar, UPI, and the India Stack "
        "APIs. Product teams often design for intermittent connectivity and multilingual "
        "interfaces because users switch between English and regional languages within a "
        "single session on low-end Android devices."
    ),
    "doc-indic-001": (
        "भारत एक विविधताओं से भरा देश है जहाँ सैकड़ों भाषाएँ बोली जाती हैं। "
        "डिजिटल साक्षरता बढ़ने के साथ हिंदी में ऑनलाइन सामग्री की माँग तेजी से बढ़ रही है। "
        "भाषा मॉडल को स्थानीय संदर्भ, सरकारी योजनाओं और सांस्कृतिक संदर्भों को समझना चाहिए।"
    ),
    "doc-indic-hi-002": (
        "मौसम विभाग ने अगले सप्ताह उत्तर भारत में हल्की बारिश की संभावना जताई है। "
        "किसानों को सलाह दी गई है कि वे फसल की कटाई का समय समायोजित करें और "
        "भंडारण स्थलों में नमी नियंत्रित रखें।"
    ),
    "doc-indic-bn-001": (
        "কলকাতায় শীতকালে তাপমাত্রা সামান্য নামলেও আর্দ্রতা বেশি থাকে। "
        "স্থানীয় বাজারে মৌসুমি সবজি ও মাছের সরবরাহ বৃদ্ধি পায়। "
        "পরিবহন ব্যবস্থা সকালে ব্যস্ত থাকায় commuters ভোরে বের হওয়ার পরামর্শ পায়।"
    ),
    "doc-indic-bn-002": (
        "অনলাইন শিক্ষা প্ল্যাটফর্মগুলো বাংলা ভাষায় কনটেন্ট বাড়াচ্ছে। "
        "শিক্ষার্থীরা ভিডিও লেকচারের পাশাপাশি অনুশীলনীর জন্য ইন্টারঅ্যাকটিভ কুইজ ব্যবহার করে।"
    ),
    "doc-indic-ta-001": (
        "சென்னையில் IT துறை வளர்ச்சி தொடர்ந்து உள்ளது. "
        "பல நிறுவனங்கள் Tamil Nadu திறமையாளர்களை பணியமர்த்துகின்றன. "
        "remote work காரணமாக suburb areas-ல் வீட்டு வ арендை அதிகரித்துள்ளது."
    ),
    "doc-indic-ta-002": (
        "தமிழ் இலக்கியத்தில் சங்க கால பாடல்கள் முக்கியமான பங்கு வகிக்கின்றன. "
        "நவீன கல்வி Tamil medium பள்ளிகளில் மொழி பாதுகாப்புடன் தொ technic திறன்களை இணைக்கிறது."
    ),
    "doc-indic-te-001": (
        "హైదరాబాద్‌లో startup ecosystem వేగంగా అభివృద్ధి చెందుతోంది. "
        "స్థానిక వ్యవసాయ మార్కెట్లలో పంటల ధరలు సీason ఆధారంగా మారుతూ ఉంటాయi."
    ),
    "doc-indic-te-002": (
        "Telugu cinema industry digital streaming platformsతో కలిసి regional content production పెంచింది. "
        "viewers short-form clips మరియు full-length releases రెండింటినీ consume చేస్తున్నారు."
    ),
    "doc-indic-mr-001": (
        "मुंबई metropolitan region मध्ये local train network दररोज millions of commuters ला सेवा देते. "
        "peak hours मध्यe crowd management आणि safety announcements महत्त्वाचे असतात."
    ),
    "doc-indic-mr-002": (
        "Maharashtra मध्यe sugarcane farming common आhe. "
        "pani vyavastha आणि market rates शetकरी निर्णयांवर थेट परिणाम करतात."
    ),
    "doc-indic-gu-001": (
        "ગુજરાતમાં small businesses textile અને diamond polishing sectors સાથે જોડાયેલા છે. "
        "GST compliance અને digital invoicing adoption gradually વધી રહ્યું છે."
    ),
    "doc-indic-kn-001": (
        "ಬೆಂಗಳೂರು tech hub ಆಗi information technology companies ಮತ್ತು numerous startups ಗe headquarters ಇದೆ. "
        "traffic congestion ಮತ್ತು metro expansion city planning ನ central topics."
    ),
    "doc-indic-ml-001": (
        "കeralaയിൽ monsoon season സമയം road maintenance crews drainage systems clear ചെയ്യുന്നു. "
        "health workers community awareness programs വഴി preventive care promote ചെയ്യുന്നു."
    ),
    "doc-anneal-001": (
        "उच्च गुणवत्ता वाला हिंदी पाठ: संविधान सभा ने भारत को sovereign democratic republic "
        "घोषित किया। मौलिक अधिकार citizens को state के arbitrary action से सुरक्षा देते हैं। "
        "यह अंश anneal चरण के लिए आरक्षित है।"
    ),
    "doc-code-001": (
        "def shard_checksum(path: Path) -> str:\n"
        "    data = path.read_bytes()\n"
        "    return hashlib.sha256(data).hexdigest()\n"
    ),
    "doc-stem-001": (
        "Problem: A train travels 120 km in 2 hours. What is its average speed? "
        "Solution: speed = distance / time = 120 / 2 = 60 km/h."
    ),
    "doc-reasoning-001": (
        "Question: If all bloops are razzies and some razzies are lazzies, can a bloop be a lazzie? "
        "Reasoning: bloops ⊆ razzies. Some razzies ∩ lazzies ≠ ∅. "
        "A bloop might map to a razzie that is not in the lazzie subset. Answer: not necessarily."
    ),
    "doc-longctx-001": (
        "Chapter excerpt: The observatory sat on a windswept ridge where night skies stayed dark "
        "enough for photometry. Engineers calibrated each sensor weekly because drift in dark "
        "current masked faint signals. Interns logged weather, humidity, and dome temperature "
        "before opening the shutter. Over months the team stitched exposures into a mosaic "
        "covering a narrow strip of the galactic plane. Reviewers asked for reproducible "
        "pipelines, versioned configs, and checksums on every intermediate file. "
        "The lead astronomer insisted that metadata travel with pixels so downstream "
        "models could audit provenance decades later."
    ),
    "doc-agentic-001": (
        '{"role":"user","content":"Find tokenizer hash for run demo"}\n'
        '{"role":"assistant","tool_calls":[{"name":"read_manifest","args":{"path":"data/tokenizer/tokenizer_manifest.json"}}]}\n'
        '{"role":"tool","content":"tokenizer_hash: tok_f841ef5bcf29"}'
    ),
    "doc-agentic-002": (
        '{"role":"user","content":"List blocked shards"}\n'
        '{"role":"assistant","tool_calls":[{"name":"query_registry","args":{"admission":"blocked"}}]}\n'
        '{"role":"tool","content":"[]"}'
    ),
    "doc-eval-001": (
        "MMLU holdout mirror: Which planet is known as the Red Planet? "
        "This row overlaps eval benchmarks and must never receive training loss."
    ),
}

EXTRA_DOCS: list[dict[str, Any]] = [
    {
        "document_id": "doc-web-004",
        "source_id": "src-toy-web",
        "title": "Distributed training overview",
        "text": (
            "Data parallelism replicates the model on multiple GPUs while splitting batches. "
            "Expert parallelism shards MoE layers so each device holds a subset of experts. "
            "All-to-all communication dispatches tokens to the experts selected by the router."
        ),
        **COMMON,
        "capability_lane": "web",
        "data_type": "pretrain",
        "packing_policy": "concat_and_chop",
        "language": "en",
        "script": "Latin",
        "never_train": False,
        "curriculum_band": "B2",
        "indic_tier": None,
        "indic_language_tier": None,
        "reasoning_trace_band": None,
        "always_on_eligible": False,
        "opus_eligible": True,
        "anneal_eligible": False,
        "epoch_budget_max": 8,
    },
    {
        "document_id": "doc-web-005",
        "source_id": "src-toy-web",
        "title": "Curriculum mixing note",
        "text": (
            "Curriculum stages ramp capability lanes at different rates. "
            "Always-on lanes maintain a floor share so core skills do not collapse "
            "when specialty lanes increase during later bands."
        ),
        **COMMON,
        "capability_lane": "web",
        "data_type": "pretrain",
        "packing_policy": "concat_and_chop",
        "language": "en",
        "script": "Latin",
        "never_train": False,
        "curriculum_band": "B1",
        "indic_tier": None,
        "indic_language_tier": None,
        "reasoning_trace_band": None,
        "always_on_eligible": True,
        "opus_eligible": True,
        "anneal_eligible": False,
        "epoch_budget_max": 8,
    },
    {
        "document_id": "doc-web-006",
        "source_id": "src-toy-web",
        "title": "Shard immutability",
        "text": (
            "Once written, a shard file is immutable. Corrections require a new shard ID "
            "and an updated manifest. Training registries reference manifests, not raw paths."
        ),
        **COMMON,
        "capability_lane": "web",
        "data_type": "pretrain",
        "packing_policy": "concat_and_chop",
        "language": "en",
        "script": "Latin",
        "never_train": False,
        "curriculum_band": "B2",
        "indic_tier": None,
        "indic_language_tier": None,
        "reasoning_trace_band": None,
        "always_on_eligible": False,
        "opus_eligible": True,
        "anneal_eligible": True,
        "epoch_budget_max": 6,
    },
    {
        "document_id": "doc-web-007",
        "source_id": "src-toy-web",
        "title": "Bangalore tech hiring trends",
        "text": (
            "Engineering teams in Bengaluru increasingly hire for ML systems roles covering "
            "data pipelines, evaluation harnesses, and inference serving. "
            "Candidates who understand tokenizer contracts and reproducible shard builds stand out."
        ),
        **COMMON,
        "capability_lane": "web",
        "data_type": "pretrain",
        "packing_policy": "concat_and_chop",
        "language": "en-IN",
        "script": "Latin",
        "never_train": False,
        "curriculum_band": "B2",
        "indic_tier": None,
        "indic_language_tier": None,
        "reasoning_trace_band": None,
        "always_on_eligible": True,
        "opus_eligible": True,
        "anneal_eligible": False,
        "epoch_budget_max": 8,
    },
    {
        "document_id": "doc-web-008",
        "source_id": "src-toy-web",
        "title": "MoE load balancing",
        "text": (
            "Router auxiliary losses encourage uniform expert utilization. "
            "Without load balancing, a few experts receive most tokens and others idle, "
            "wasting capacity and destabilizing gradients."
        ),
        **COMMON,
        "capability_lane": "web",
        "data_type": "pretrain",
        "packing_policy": "concat_and_chop",
        "language": "en",
        "script": "Latin",
        "never_train": False,
        "curriculum_band": "B3",
        "indic_tier": None,
        "indic_language_tier": None,
        "reasoning_trace_band": None,
        "always_on_eligible": False,
        "opus_eligible": True,
        "anneal_eligible": True,
        "epoch_budget_max": 4,
    },
    {
        "document_id": "doc-web-009",
        "source_id": "src-toy-web",
        "title": "Run ledger semantics",
        "text": (
            "Consumption ledgers record which shard windows were trained at each step. "
            "Crash recovery replays from the last committed checkpoint plus ledger cursor."
        ),
        **COMMON,
        "capability_lane": "web",
        "data_type": "pretrain",
        "packing_policy": "concat_and_chop",
        "language": "en",
        "script": "Latin",
        "never_train": False,
        "curriculum_band": "B2",
        "indic_tier": None,
        "indic_language_tier": None,
        "reasoning_trace_band": None,
        "always_on_eligible": False,
        "opus_eligible": True,
        "anneal_eligible": False,
        "epoch_budget_max": 8,
    },
    {
        "document_id": "doc-indic-hi-003",
        "source_id": "src-toy-indic",
        "title": "Hindi agriculture bulletin",
        "text": (
            "सिंचाई के लिए सौर pump subsidy योजना का लाभ उठाने हेतु किसान ऑनलाइन आवेदन कर सकते हैं। "
            "जिला कृषि अधिकारी कार्यालय सत्यापन के बाद अनुदान जारी करता है।"
        ),
        **COMMON,
        "capability_lane": "indic",
        "data_type": "pretrain",
        "packing_policy": "concat_and_chop",
        "language": "hi",
        "script": "Devanagari",
        "never_train": False,
        "curriculum_band": "B1",
        "indic_tier": "A",
        "indic_language_tier": "T1",
        "reasoning_trace_band": None,
        "always_on_eligible": True,
        "opus_eligible": True,
        "anneal_eligible": False,
        "epoch_budget_max": 8,
    },
    {
        "document_id": "doc-indic-bn-003",
        "source_id": "src-toy-indic",
        "title": "Bengali health camp notice",
        "text": (
            "পাড়া ক্লিনিকে বিনামূল্যে রক্তচাপ পরীক্ষার camp আগামী শনিবার সকাল ৯টা থেকে চলবে। "
            "বয়স্ক residents দের জন্য priority queue থাকবে।"
        ),
        **COMMON,
        "capability_lane": "indic",
        "data_type": "pretrain",
        "packing_policy": "concat_and_chop",
        "language": "bn",
        "script": "Bengali",
        "never_train": False,
        "curriculum_band": "B1",
        "indic_tier": "A",
        "indic_language_tier": "T1",
        "reasoning_trace_band": None,
        "always_on_eligible": True,
        "opus_eligible": True,
        "anneal_eligible": False,
        "epoch_budget_max": 8,
    },
    {
        "document_id": "doc-indic-ta-003",
        "source_id": "src-toy-indic",
        "title": "Tamil civic services",
        "text": (
            "Chennai Corporation online portal மூலம் property tax payment செய்யலாம். "
            "receipt email மற்றும் SMS இரண்டு channels-லும் அனுப்பப்படும்."
        ),
        **COMMON,
        "capability_lane": "indic",
        "data_type": "pretrain",
        "packing_policy": "concat_and_chop",
        "language": "ta",
        "script": "Tamil",
        "never_train": False,
        "curriculum_band": "B2",
        "indic_tier": "B",
        "indic_language_tier": "T1",
        "reasoning_trace_band": None,
        "always_on_eligible": False,
        "opus_eligible": True,
        "anneal_eligible": False,
        "epoch_budget_max": 8,
    },
    {
        "document_id": "doc-indic-gu-002",
        "source_id": "src-toy-indic",
        "title": "Gujarati festival logistics",
        "text": (
            "Navratri દરમિયાન cities માં traffic diversions announced થાય છે. "
            "public transport extended hours offer करवामાં આવે છે."
        ),
        **COMMON,
        "capability_lane": "indic",
        "data_type": "pretrain",
        "packing_policy": "concat_and_chop",
        "language": "gu",
        "script": "Gujarati",
        "never_train": False,
        "curriculum_band": "B2",
        "indic_tier": "B",
        "indic_language_tier": "T2",
        "reasoning_trace_band": None,
        "always_on_eligible": False,
        "opus_eligible": True,
        "anneal_eligible": False,
        "epoch_budget_max": 6,
    },
    {
        "document_id": "doc-code-002",
        "source_id": "src-toy-code",
        "title": "Router top-k selection",
        "text": (
            "def topk_route(logits: Tensor, k: int) -> tuple[Tensor, Tensor]:\n"
            "    probs = torch.softmax(logits, dim=-1)\n"
            "    weights, indices = torch.topk(probs, k, dim=-1)\n"
            "    return indices, weights / weights.sum(dim=-1, keepdim=True)\n"
        ),
        **COMMON,
        "capability_lane": "code",
        "data_type": "pretrain",
        "packing_policy": "concat_and_chop",
        "language": "en",
        "script": "Latin",
        "never_train": False,
        "curriculum_band": "B2",
        "indic_tier": None,
        "indic_language_tier": None,
        "reasoning_trace_band": None,
        "always_on_eligible": False,
        "opus_eligible": True,
        "anneal_eligible": True,
        "epoch_budget_max": 6,
    },
    {
        "document_id": "doc-code-003",
        "source_id": "src-toy-code",
        "title": "Atomic shard write",
        "text": (
            "def write_shard_atomic(path: Path, payload: bytes) -> None:\n"
            "    tmp = path.with_suffix(path.suffix + '.tmp')\n"
            "    tmp.write_bytes(payload)\n"
            "    tmp.replace(path)\n"
        ),
        **COMMON,
        "capability_lane": "code",
        "data_type": "pretrain",
        "packing_policy": "concat_and_chop",
        "language": "en",
        "script": "Latin",
        "never_train": False,
        "curriculum_band": "B2",
        "indic_tier": None,
        "indic_language_tier": None,
        "reasoning_trace_band": None,
        "always_on_eligible": False,
        "opus_eligible": True,
        "anneal_eligible": False,
        "epoch_budget_max": 8,
    },
    {
        "document_id": "doc-code-004",
        "source_id": "src-toy-code",
        "title": "Manifest validation snippet",
        "text": (
            "class ShardManifest(BaseModel):\n"
            "    shard_id: str\n"
            "    tokenizer_hash: str\n"
            "    token_count: int\n"
            "    content_hash: str\n"
        ),
        **COMMON,
        "capability_lane": "code",
        "data_type": "pretrain",
        "packing_policy": "concat_and_chop",
        "language": "en",
        "script": "Latin",
        "never_train": False,
        "curriculum_band": "B3",
        "indic_tier": None,
        "indic_language_tier": None,
        "reasoning_trace_band": None,
        "always_on_eligible": False,
        "opus_eligible": True,
        "anneal_eligible": True,
        "epoch_budget_max": 4,
    },
    {
        "document_id": "doc-code-005",
        "source_id": "src-toy-code",
        "title": "Rust tokenizer binding stub",
        "text": (
            "pub fn encode_batch(texts: &[&str], tokenizer: &Tokenizer) -> Vec<Vec<u32>> {\n"
            "    texts.iter().map(|t| tokenizer.encode(t).ids).collect()\n"
            "}\n"
        ),
        **COMMON,
        "capability_lane": "code",
        "data_type": "pretrain",
        "packing_policy": "concat_and_chop",
        "language": "en",
        "script": "Latin",
        "never_train": False,
        "curriculum_band": "B2",
        "indic_tier": None,
        "indic_language_tier": None,
        "reasoning_trace_band": None,
        "always_on_eligible": False,
        "opus_eligible": True,
        "anneal_eligible": False,
        "epoch_budget_max": 6,
    },
    {
        "document_id": "doc-code-006",
        "source_id": "src-toy-code",
        "title": "SQL experiment registry",
        "text": (
            "CREATE TABLE training_runs (\n"
            "  run_id TEXT PRIMARY KEY,\n"
            "  tokenizer_hash TEXT NOT NULL,\n"
            "  started_at TIMESTAMPTZ NOT NULL\n"
            ");\n"
        ),
        **COMMON,
        "capability_lane": "code",
        "data_type": "pretrain",
        "packing_policy": "concat_and_chop",
        "language": "en",
        "script": "Latin",
        "never_train": False,
        "curriculum_band": "B2",
        "indic_tier": None,
        "indic_language_tier": None,
        "reasoning_trace_band": None,
        "always_on_eligible": False,
        "opus_eligible": True,
        "anneal_eligible": False,
        "epoch_budget_max": 8,
    },
    {
        "document_id": "doc-stem-002",
        "source_id": "src-toy-web",
        "title": "Physics kinematics",
        "text": (
            "A ball is dropped from rest from height h. Using g = 9.8 m/s^2, "
            "time to ground is t = sqrt(2h/g). For h = 20 m, t ≈ 2.02 s."
        ),
        **COMMON,
        "capability_lane": "stem",
        "data_type": "pretrain",
        "packing_policy": "concat_and_chop",
        "language": "en",
        "script": "Latin",
        "never_train": False,
        "curriculum_band": "B2",
        "indic_tier": None,
        "indic_language_tier": None,
        "reasoning_trace_band": None,
        "always_on_eligible": False,
        "opus_eligible": True,
        "anneal_eligible": True,
        "epoch_budget_max": 6,
    },
    {
        "document_id": "doc-stem-003",
        "source_id": "src-toy-web",
        "title": "Chemistry stoichiometry",
        "text": (
            "Balance: CH4 + 2 O2 -> CO2 + 2 H2O. One mole methane consumes two moles oxygen "
            "and produces one mole carbon dioxide plus two moles water."
        ),
        **COMMON,
        "capability_lane": "stem",
        "data_type": "pretrain",
        "packing_policy": "concat_and_chop",
        "language": "en",
        "script": "Latin",
        "never_train": False,
        "curriculum_band": "B2",
        "indic_tier": None,
        "indic_language_tier": None,
        "reasoning_trace_band": None,
        "always_on_eligible": False,
        "opus_eligible": True,
        "anneal_eligible": False,
        "epoch_budget_max": 8,
    },
    {
        "document_id": "doc-stem-004",
        "source_id": "src-toy-web",
        "title": "Biology cell division",
        "text": (
            "Mitosis produces two genetically identical daughter cells for growth and repair. "
            "Meiosis reduces chromosome number by half to form gametes for sexual reproduction."
        ),
        **COMMON,
        "capability_lane": "stem",
        "data_type": "pretrain",
        "packing_policy": "concat_and_chop",
        "language": "en",
        "script": "Latin",
        "never_train": False,
        "curriculum_band": "B3",
        "indic_tier": None,
        "indic_language_tier": None,
        "reasoning_trace_band": None,
        "always_on_eligible": False,
        "opus_eligible": True,
        "anneal_eligible": True,
        "epoch_budget_max": 4,
    },
    {
        "document_id": "doc-stem-005",
        "source_id": "src-toy-web",
        "title": "Probability basics",
        "text": (
            "For a fair coin, P(heads) = 0.5. Two independent flips yield four equally likely "
            "outcomes. P(at least one heads) = 1 - P(both tails) = 0.75."
        ),
        **COMMON,
        "capability_lane": "stem",
        "data_type": "pretrain",
        "packing_policy": "concat_and_chop",
        "language": "en",
        "script": "Latin",
        "never_train": False,
        "curriculum_band": "B2",
        "indic_tier": None,
        "indic_language_tier": None,
        "reasoning_trace_band": None,
        "always_on_eligible": False,
        "opus_eligible": True,
        "anneal_eligible": False,
        "epoch_budget_max": 6,
    },
    {
        "document_id": "doc-stem-006",
        "source_id": "src-toy-web",
        "title": "Calculus derivative",
        "text": (
            "The derivative of f(x) = x^3 is f'(x) = 3x^2. At x = 2, the tangent slope is 12."
        ),
        **COMMON,
        "capability_lane": "stem",
        "data_type": "pretrain",
        "packing_policy": "concat_and_chop",
        "language": "en",
        "script": "Latin",
        "never_train": False,
        "curriculum_band": "B3",
        "indic_tier": None,
        "indic_language_tier": None,
        "reasoning_trace_band": None,
        "always_on_eligible": False,
        "opus_eligible": True,
        "anneal_eligible": True,
        "epoch_budget_max": 4,
    },
    {
        "document_id": "doc-reasoning-002",
        "source_id": "src-toy-web",
        "title": "Medium reasoning: scheduling",
        "text": (
            "Three meetings A, B, C must occur in order A before B before C. "
            "A takes 30 minutes starting at 9:00. B needs 45 minutes after A ends. "
            "C needs 20 minutes. Earliest finish: 9:00 + 30 + 45 + 20 = 10:35."
        ),
        **COMMON,
        "capability_lane": "reasoning",
        "data_type": "pretrain",
        "packing_policy": "concat_and_chop",
        "language": "en",
        "script": "Latin",
        "never_train": False,
        "curriculum_band": "B3",
        "indic_tier": None,
        "indic_language_tier": None,
        "reasoning_trace_band": "medium",
        "always_on_eligible": True,
        "opus_eligible": True,
        "anneal_eligible": False,
        "epoch_budget_max": 4,
    },
    {
        "document_id": "doc-reasoning-003",
        "source_id": "src-toy-web",
        "title": "Long reasoning: grid path",
        "text": (
            "On a 3x3 grid from top-left to bottom-right moving only right or down, "
            "count paths. Row 1: 1. Row 2: 1,2. Row 3: 1,3,6. Answer: 6 paths. "
            "General formula for nxn grid is binomial(2n-2, n-1)."
        ),
        **COMMON,
        "capability_lane": "reasoning",
        "data_type": "pretrain",
        "packing_policy": "concat_and_chop",
        "language": "en",
        "script": "Latin",
        "never_train": False,
        "curriculum_band": "B4",
        "indic_tier": None,
        "indic_language_tier": None,
        "reasoning_trace_band": "long",
        "always_on_eligible": False,
        "opus_eligible": True,
        "anneal_eligible": True,
        "epoch_budget_max": 3,
    },
    {
        "document_id": "doc-reasoning-004",
        "source_id": "src-toy-web",
        "title": "Short reasoning: parity",
        "text": (
            "Sum of two odd integers is even because (2a+1)+(2b+1)=2(a+b+1)."
        ),
        **COMMON,
        "capability_lane": "reasoning",
        "data_type": "pretrain",
        "packing_policy": "concat_and_chop",
        "language": "en",
        "script": "Latin",
        "never_train": False,
        "curriculum_band": "B2",
        "indic_tier": None,
        "indic_language_tier": None,
        "reasoning_trace_band": "short",
        "always_on_eligible": True,
        "opus_eligible": True,
        "anneal_eligible": False,
        "epoch_budget_max": 6,
    },
    {
        "document_id": "doc-reasoning-005",
        "source_id": "src-toy-web",
        "title": "Ultra reasoning: invariant",
        "text": (
            "Consider tokens routed to experts each step. Load balancing loss uses "
            "fraction of tokens per expert vs router probability mass. "
            "If router collapses to one expert, auxiliary loss spikes, gradient pushes "
            "probabilities toward uniform until utilization variance drops."
        ),
        **COMMON,
        "capability_lane": "reasoning",
        "data_type": "pretrain",
        "packing_policy": "concat_and_chop",
        "language": "en",
        "script": "Latin",
        "never_train": False,
        "curriculum_band": "B4",
        "indic_tier": None,
        "indic_language_tier": None,
        "reasoning_trace_band": "ultra",
        "always_on_eligible": False,
        "opus_eligible": True,
        "anneal_eligible": True,
        "epoch_budget_max": 2,
    },
    {
        "document_id": "doc-longctx-002",
        "source_id": "src-toy-web",
        "title": "Long context: policy memo",
        "text": (
            "Internal memo draft: The training data execution system must treat tokenizer "
            "identity as immutable for a run family. Shards reference tokenizer_hash in "
            "every manifest row. Admission gates reject documents missing cleaning lineage "
            "or with eval overlap flags. Curriculum YAML defines stage bands; the sampler "
            "reads registry entries marked opus_eligible. Anneal reserves high-quality indic "
            "text for late-stage loss weighting. Firewall checks run before batch tensors "
            "receive loss masks. Never-train rows remain in the registry for overlap detection "
            "but never enter consumption ledgers. Crash recovery depends on atomic shard writes "
            "and append-only ledgers keyed by run_id and step."
        ),
        **COMMON,
        "capability_lane": "long_context",
        "data_type": "pretrain",
        "packing_policy": "concat_and_chop",
        "language": "en",
        "script": "Latin",
        "never_train": False,
        "curriculum_band": "B3",
        "indic_tier": None,
        "indic_language_tier": None,
        "reasoning_trace_band": None,
        "always_on_eligible": False,
        "opus_eligible": True,
        "anneal_eligible": True,
        "epoch_budget_max": 3,
    },
    {
        "document_id": "doc-longctx-003",
        "source_id": "src-toy-web",
        "title": "Long context: travelogue",
        "text": (
            "Day 1: Departed Mumbai Central by train toward Pune, reading notes on shard "
            "packing policies. Day 2: Visited a coworking space where engineers debated "
            "concat_and_chop versus document-boundary packing for code lanes. Day 3: "
            "Returned via express highway documenting latency tradeoffs for streaming "
            "tokenizers. Each evening I logged char counts and checksums in a travel "
            "journal to mimic manifest discipline."
        ),
        **COMMON,
        "capability_lane": "long_context",
        "data_type": "pretrain",
        "packing_policy": "concat_and_chop",
        "language": "en",
        "script": "Latin",
        "never_train": False,
        "curriculum_band": "B4",
        "indic_tier": None,
        "indic_language_tier": None,
        "reasoning_trace_band": None,
        "always_on_eligible": False,
        "opus_eligible": True,
        "anneal_eligible": False,
        "epoch_budget_max": 2,
    },
    {
        "document_id": "doc-agentic-003",
        "source_id": "src-toy-agentic",
        "title": "Agentic: validate curriculum",
        "text": (
            '{"role":"user","content":"Validate curriculum.yaml shares sum to 1"}\n'
            '{"role":"assistant","tool_calls":[{"name":"run_validator","args":{"path":"configs/curriculum.yaml"}}]}\n'
            '{"role":"tool","content":"ok: stage S1 shares sum 1.0"}'
        ),
        **COMMON,
        "capability_lane": "agentic",
        "data_type": "agentic",
        "packing_policy": "structure_preserving",
        "language": "en",
        "script": "Latin",
        "never_train": False,
        "curriculum_band": "B2",
        "indic_tier": None,
        "indic_language_tier": None,
        "reasoning_trace_band": None,
        "always_on_eligible": True,
        "opus_eligible": True,
        "anneal_eligible": False,
        "epoch_budget_max": 4,
    },
    {
        "document_id": "doc-agentic-004",
        "source_id": "src-toy-agentic",
        "title": "Agentic: build shards",
        "text": (
            '{"role":"user","content":"Build shards for demo config"}\n'
            '{"role":"assistant","tool_calls":[{"name":"build_shards","args":{"config":"configs/demo.yaml"}}]}\n'
            '{"role":"tool","content":"wrote 3 shards, manifest shard_manifest.json"}'
        ),
        **COMMON,
        "capability_lane": "agentic",
        "data_type": "agentic",
        "packing_policy": "structure_preserving",
        "language": "en",
        "script": "Latin",
        "never_train": False,
        "curriculum_band": "B3",
        "indic_tier": None,
        "indic_language_tier": None,
        "reasoning_trace_band": None,
        "always_on_eligible": False,
        "opus_eligible": True,
        "anneal_eligible": False,
        "epoch_budget_max": 4,
    },
    {
        "document_id": "doc-agentic-005",
        "source_id": "src-toy-agentic",
        "title": "Agentic: admission check",
        "text": (
            '{"role":"user","content":"Why was shard-007 blocked?"}\n'
            '{"role":"assistant","tool_calls":[{"name":"inspect_admission","args":{"shard_id":"shard-007"}}]}\n'
            '{"role":"tool","content":"blocked: missing tokenizer_hash"}'
        ),
        **COMMON,
        "capability_lane": "agentic",
        "data_type": "agentic",
        "packing_policy": "structure_preserving",
        "language": "en",
        "script": "Latin",
        "never_train": False,
        "curriculum_band": "B2",
        "indic_tier": None,
        "indic_language_tier": None,
        "reasoning_trace_band": None,
        "always_on_eligible": True,
        "opus_eligible": True,
        "anneal_eligible": False,
        "epoch_budget_max": 4,
    },
    {
        "document_id": "doc-agentic-006",
        "source_id": "src-toy-agentic",
        "title": "Agentic: resume run",
        "text": (
            '{"role":"user","content":"Resume training from last checkpoint"}\n'
            '{"role":"assistant","tool_calls":[{"name":"resume_run","args":{"run_id":"run-demo-001"}}]}\n'
            '{"role":"tool","content":"resumed at step 1200 from ledger cursor"}'
        ),
        **COMMON,
        "capability_lane": "agentic",
        "data_type": "agentic",
        "packing_policy": "structure_preserving",
        "language": "en",
        "script": "Latin",
        "never_train": False,
        "curriculum_band": "B3",
        "indic_tier": None,
        "indic_language_tier": None,
        "reasoning_trace_band": None,
        "always_on_eligible": False,
        "opus_eligible": True,
        "anneal_eligible": False,
        "epoch_budget_max": 4,
    },
    {
        "document_id": "doc-agentic-007",
        "source_id": "src-toy-agentic",
        "title": "Agentic: firewall audit",
        "text": (
            '{"role":"user","content":"Show firewall blocks in last run"}\n'
            '{"role":"assistant","tool_calls":[{"name":"grep_run_log","args":{"pattern":"firewall_block"}}]}\n'
            '{"role":"tool","content":"1 block: eval shard doc-eval-001"}'
        ),
        **COMMON,
        "capability_lane": "agentic",
        "data_type": "agentic",
        "packing_policy": "structure_preserving",
        "language": "en",
        "script": "Latin",
        "never_train": False,
        "curriculum_band": "B3",
        "indic_tier": None,
        "indic_language_tier": None,
        "reasoning_trace_band": None,
        "always_on_eligible": False,
        "opus_eligible": True,
        "anneal_eligible": False,
        "epoch_budget_max": 4,
    },
    {
        # Deliberate contamination for the firewall demo (P4-T04).
        #
        # Its metadata says clean: eval_overlap_status "clear", never_train False. That
        # is the point. Upstream cleaning missed that the body mirrors an MMLU holdout
        # item, so this document is admitted, enters the sample pool, and gets planned
        # like any other web row. Only the firewall's canary check on the actual text
        # catches it, which is exactly the leak the firewall exists for.
        #
        # Marking it never_train instead would exclude it at the pool and prove nothing:
        # the demo would show the metadata filter working, not the firewall.
        "document_id": "doc-web-contaminated-001",
        "source_id": "src-toy-web",
        "title": "Planetary science quiz notes",
        "text": (
            "Study notes scraped from a quiz site. Which planet is known as the Red "
            "Planet? Iron oxide on the surface gives Mars its colour, and the same "
            "dust drives the global storms that can obscure the whole disc for weeks."
        ),
        **COMMON,
        "capability_lane": "web",
        "data_type": "pretrain",
        "packing_policy": "concat_and_chop",
        "language": "en",
        "script": "Latin",
        "never_train": False,
        "curriculum_band": "B1",
        "indic_tier": None,
        "indic_language_tier": None,
        "reasoning_trace_band": None,
        "always_on_eligible": False,
        "opus_eligible": True,
        "anneal_eligible": False,
        "epoch_budget_max": 4,
    },
]


def finalize(record: dict[str, Any]) -> dict[str, Any]:
    text = record.pop("text", None)
    if text is None:
        text = TEXT[record["document_id"]]
    record["text"] = text
    record["content_status"] = "ready"
    record["char_count"] = len(text)
    record["content_sha256"] = content_sha256(text)
    return record


def main() -> None:
    corpus_dir = _ASSIGNMENT / "data" / "toy_corpus"
    _, skeleton = load_corpus(corpus_dir)

    by_id = {doc["document_id"]: doc for doc in skeleton}
    for doc_id, text in TEXT.items():
        if doc_id not in by_id:
            raise SystemExit(f"TEXT entry {doc_id} not in skeleton")
        by_id[doc_id] = finalize({**by_id[doc_id], "text": text})

    # Re-runnable: the skeleton is read from documents.jsonl, which already contains the
    # extras added by earlier runs. EXTRA_DOCS is the source of truth for those rows, so
    # they are rebuilt in place; only a genuine duplicate *within* EXTRA_DOCS is an error.
    extra_ids: list[str] = []
    for extra in EXTRA_DOCS:
        doc_id = extra["document_id"]
        if doc_id in extra_ids:
            raise SystemExit(f"duplicate document_id in EXTRA_DOCS: {doc_id}")
        extra_ids.append(doc_id)
        by_id[doc_id] = finalize(dict(extra))

    provenance, _ = load_corpus(corpus_dir)
    prov_ids = set(provenance.keys())
    ordered_ids = [doc["document_id"] for doc in skeleton]
    ordered_ids += [doc_id for doc_id in extra_ids if doc_id not in set(ordered_ids)]
    documents = [by_id[doc_id] for doc_id in ordered_ids]

    for doc in documents:
        validate_document_record(doc, provenance_ids=prov_ids)

    out_path = corpus_dir / "documents.jsonl"
    with out_path.open("w", encoding="utf-8", newline="\n") as handle:
        for doc in documents:
            handle.write(json.dumps(doc, ensure_ascii=False) + "\n")

    print(f"Wrote {len(documents)} documents to {out_path}")


if __name__ == "__main__":
    main()
