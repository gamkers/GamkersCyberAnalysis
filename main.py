import streamlit as st
from st_on_hover_tabs import on_hover_tabs
from typing import List, Dict, Tuple
from langchain_groq import ChatGroq
from langchain_classic.chains import LLMChain, ConversationChain
from langchain_classic.prompts import PromptTemplate
from langchain_classic.memory import ConversationBufferMemory
import json, math, struct, re, os, subprocess, tempfile
from collections import Counter
from docx import Document
from streamlit_lottie import st_lottie

# ── Hide Streamlit chrome ──────────────────────────────────────────────────────
st.markdown("<style>MainMenu{visibility:hidden;}footer{visibility:hidden;}</style>", unsafe_allow_html=True)
try:
    with open("style.css") as f:
        st.markdown('<style>' + f.read() + '</style>', unsafe_allow_html=True)
except Exception:
    pass

dark_purple_theme = """
<style>
    :root{--primary-color:#7B2CBF;--background-color:black;--secondary-bg:#2D2D2D;--text-color:#FFFFFF;--accent-color:#9D4EDD;}
    .stApp{background-color:var(--background-color);color:var(--text-color);}
    h1,h2,h3{color:var(--accent-color)!important;}
    .stButton>button{background-color:var(--primary-color);color:white;border:none;border-radius:4px;transition:all 0.3s ease;}
    .stButton>button:hover{background-color:var(--accent-color);transform:translateY(-2px);}
    .stTextArea>div>div>textarea{background-color:var(--secondary-bg);color:var(--text-color);border:1px solid var(--primary-color);}
    .stFileUploader{background-color:var(--secondary-bg);border:1px dashed var(--primary-color);border-radius:4px;}
    .stProgress>div>div>div>div{background-color:var(--primary-color);}
    .stChatMessage{background-color:var(--secondary-bg);border-radius:8px;padding:10px;margin:5px 0;}
    .alert-box{background:#1a0a2e;border:1px solid #7B2CBF;border-radius:8px;padding:12px;margin:8px 0;}
    .threat-high{border-left:4px solid #ff4444;}
    .threat-med{border-left:4px solid #ffaa00;}
    .threat-low{border-left:4px solid #44ff88;}
</style>
"""
st.markdown(dark_purple_theme, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# BINARY FORENSICS ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class BinaryForensics:
    """Static analysis helpers that run before LLM analysis."""

    # Known msfvenom / Metasploit shellcode byte sequences (first ~8 bytes as hex)
    SHELLCODE_SIGS = [
        b"\xfc\xe8\x82\x00\x00\x00",   # meterpreter x86 prologue
        b"\xfc\xe8\x89\x00\x00\x00",   # meterpreter x86 variant
        b"\xfc\x48\x83\xe4\xf0\xe8",   # meterpreter x64 prologue
        b"\xd9\xeb\x9b\xd9\x74\x24",   # shikata_ga_nai encoder stub
        b"\x31\xc9\x83\xe9",           # XOR decode loop
        b"\x60\x89\xe5\x31\xd2",       # egghunter pattern
        b"\xeb\x27\x5e\x89\x76",       # jump-call-pop shellcode
    ]

    # Suspicious import DLLs msfvenom payloads commonly resolve
    SUSPICIOUS_IMPORTS = {
        "WS2_32.dll", "WININET.dll", "URLMON.dll",
        "WINHTTP.dll", "NTDLL.dll", "MSVCRT.dll",
    }

    # High-risk API names resolved by shellcode stagers
    STAGER_APIS = [
        "VirtualAlloc", "VirtualAllocEx", "VirtualProtect",
        "WriteProcessMemory", "CreateRemoteThread", "NtCreateThreadEx",
        "RtlCreateUserThread", "LoadLibraryA", "LoadLibraryW",
        "GetProcAddress", "ShellExecuteA", "ShellExecuteW",
        "WinExec", "CreateProcessA", "CreateProcessW",
        "URLDownloadToFile", "InternetOpenUrl", "InternetOpen",
        "HttpSendRequest", "WSAStartup", "WSASocket", "connect",
        "recv", "send", "OpenProcess", "TerminateProcess",
        "SetWindowsHookEx", "GetAsyncKeyState",           # keylogger
        "CryptEncrypt", "CryptDecrypt",                    # ransomware
        "RegSetValueEx", "RegCreateKeyEx",                 # persistence
        "CreateService", "OpenSCManager",                  # service install
        "IsDebuggerPresent", "CheckRemoteDebuggerPresent", # anti-debug
        "GetTickCount", "QueryPerformanceCounter",          # timing checks
        "NtQueryInformationProcess",                        # anti-analysis
    ]

    # Encoder / packer artifacts found as strings
    ENCODER_ARTIFACTS = [
        "shikata", "EXITFUNC", "staged", "stager",
        "metsrv", "meterpreter", "ReflectiveDll",
        "payload", "shellcode", "exploit", "msf",
        "reverse_tcp", "reverse_https", "bind_tcp",
        "migrate", "getsystem", "hashdump",
    ]

    @staticmethod
    def shannon_entropy(data: bytes) -> float:
        if not data:
            return 0.0
        counts = Counter(data)
        length = len(data)
        return -sum((c / length) * math.log2(c / length) for c in counts.values())

    @staticmethod
    def section_entropy(data: bytes, section_size: int = 4096) -> List[Tuple[int, float]]:
        results = []
        for offset in range(0, len(data), section_size):
            chunk = data[offset:offset + section_size]
            if len(chunk) < 64:
                continue
            ent = BinaryForensics.shannon_entropy(chunk)
            results.append((offset, ent))
        return results

    @staticmethod
    def parse_pe_header(data: bytes) -> Dict:
        result = {"is_pe": False, "imports": [], "sections": [], "characteristics": []}
        try:
            if data[:2] != b"MZ":
                return result
            result["is_pe"] = True
            pe_offset = struct.unpack_from("<I", data, 0x3C)[0]
            if pe_offset + 24 > len(data):
                return result
            if data[pe_offset:pe_offset+4] != b"PE\x00\x00":
                return result
            machine = struct.unpack_from("<H", data, pe_offset + 4)[0]
            result["architecture"] = {0x14c: "x86 (32-bit)", 0x8664: "x86-64 (64-bit)"}.get(machine, f"Unknown (0x{machine:04x})")
            characteristics = struct.unpack_from("<H", data, pe_offset + 22)[0]
            if characteristics & 0x0002: result["characteristics"].append("Executable")
            if characteristics & 0x2000: result["characteristics"].append("DLL")
            if characteristics & 0x0020: result["characteristics"].append("Large address aware")
        except Exception:
            pass
        return result

    @staticmethod
    def find_shellcode_signatures(data: bytes) -> List[str]:
        found = []
        for sig in BinaryForensics.SHELLCODE_SIGS:
            offset = data.find(sig)
            if offset != -1:
                found.append(f"Known shellcode signature at offset 0x{offset:08X}: {sig.hex()}")
        return found

    @staticmethod
    def find_suspicious_strings(raw_strings: List[str]) -> Dict[str, List[str]]:
        findings = {
            "stager_apis": [],
            "encoder_artifacts": [],
            "network_iocs": [],
            "registry_persistence": [],
            "process_injection": [],
            "anti_analysis": [],
            "encoded_blobs": [],
        }

        ip_pattern = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')
        url_pattern = re.compile(r'https?://[^\s"\'<>]+', re.IGNORECASE)
        b64_pattern = re.compile(r'[A-Za-z0-9+/]{40,}={0,2}')
        hex_blob_pattern = re.compile(r'(?:[0-9a-fA-F]{2}){16,}')

        for s in raw_strings:
            sl = s.lower()
            # Stager APIs
            for api in BinaryForensics.STAGER_APIS:
                if api.lower() in sl:
                    findings["stager_apis"].append(f"{api} → {s[:80]}")

            # Encoder/payload artifacts
            for artifact in BinaryForensics.ENCODER_ARTIFACTS:
                if artifact.lower() in sl:
                    findings["encoder_artifacts"].append(f"{artifact} → {s[:80]}")

            # Network IOCs
            for ip in ip_pattern.findall(s):
                findings["network_iocs"].append(f"IP address: {ip}")
            for url in url_pattern.findall(s):
                findings["network_iocs"].append(f"URL: {url}")

            # Registry persistence
            if any(k in sl for k in ["software\\microsoft\\windows\\currentversion\\run",
                                      "hkey_local_machine", "hkey_current_user", "regsetvalue"]):
                findings["registry_persistence"].append(s[:100])

            # Process injection
            if any(k in sl for k in ["virtualalloc", "writeprocessmemory",
                                      "createremotethread", "openprocess"]):
                findings["process_injection"].append(s[:100])

            # Anti-analysis
            if any(k in sl for k in ["isdebuggerpresent", "checkremotedebugger",
                                      "ntqueryinformation", "vmware", "virtualbox",
                                      "sandboxie", "wireshark", "procmon"]):
                findings["anti_analysis"].append(s[:100])

            # Encoded blobs
            if b64_pattern.search(s):
                findings["encoded_blobs"].append(f"Possible base64: {s[:60]}...")
            if hex_blob_pattern.search(s):
                findings["encoded_blobs"].append(f"Possible hex blob: {s[:60]}...")

        # Deduplicate
        for k in findings:
            findings[k] = list(dict.fromkeys(findings[k]))[:15]

        return findings

    @classmethod
    def full_analysis(cls, data: bytes) -> Dict:
        """Run all static checks and return a structured forensics report."""
        report = {}

        # Entropy
        overall_entropy = cls.shannon_entropy(data)
        report["overall_entropy"] = round(overall_entropy, 3)
        report["entropy_verdict"] = (
            "🔴 HIGH — packed/encrypted/shellcode likely" if overall_entropy > 7.0 else
            "🟡 MEDIUM — possible compression or encoding" if overall_entropy > 6.0 else
            "🟢 LOW — mostly plaintext/readable content"
        )

        # High-entropy sections
        sections = cls.section_entropy(data)
        high_ent_sections = [(off, round(e, 2)) for off, e in sections if e > 7.0]
        report["high_entropy_sections"] = [
            f"Offset 0x{off:08X}: entropy={ent}" for off, ent in high_ent_sections[:10]
        ]

        # PE header
        report["pe_info"] = cls.parse_pe_header(data)

        # Shellcode signatures
        report["shellcode_signatures"] = cls.find_shellcode_signatures(data)

        # Null-byte / shellcode NOP sled detection
        nop_sled = data.find(b"\x90" * 16)
        report["nop_sled_detected"] = nop_sled != -1
        if nop_sled != -1:
            report["nop_sled_offset"] = f"0x{nop_sled:08X}"

        # EXITFUNC marker (msfvenom specific)
        exitfunc_hits = [m.start() for m in re.finditer(b"EXITFUNC", data)]
        report["exitfunc_markers"] = [f"0x{o:08X}" for o in exitfunc_hits]

        # Suspicious import strings
        import_hits = [imp for imp in cls.SUSPICIOUS_IMPORTS if imp.encode() in data]
        report["suspicious_imports"] = import_hits

        return report


# ══════════════════════════════════════════════════════════════════════════════
# API KEY SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

def api_key_sidebar() -> str:
    with st.sidebar:
        st.markdown("### 🔑 Groq API Key")
        key = st.text_input(
            "Enter your Groq API key",
            type="password",
            placeholder="gsk_...",
            help="Never stored — lives only in this browser session.",
        )
        if key:
            st.success("Key loaded ✓", icon="✅")
        else:
            st.warning("Paste your Groq API key to get started.")
    return key


# ══════════════════════════════════════════════════════════════════════════════
# ABOUT
# ══════════════════════════════════════════════════════════════════════════════

def load_lottie_urls():
    return {
        "shield":   "https://assets5.lottiefiles.com/packages/lf20_yom6uvgj.json",
        "analysis": "https://assets8.lottiefiles.com/packages/lf20_qmfs6c3i.json",
        "chat":     "https://assets8.lottiefiles.com/packages/lf20_2LdLki.json",
        "security": "https://assets8.lottiefiles.com/packages/lf20_oyi9a28g.json",
    }

def about_section():
    animations = load_lottie_urls()
    st.title("GAMKERS Security Analysis Suite")
    st.write("---")
    with st.container():
        l, r = st.columns(2)
        with l:
            st.header("Advanced Security Analysis")
            st.write("AI-powered deep code inspection and vulnerability detection.")
            st.button("🚀 Try Analysis Now", key="try_analysis")
        with r:
            st_lottie(animations["analysis"], height=300, key="analysis_anim")
    st.write("---")
    with st.container():
        l, r = st.columns(2)
        with l:
            st_lottie(animations["chat"], height=300, key="chat_anim")
        with r:
            st.header("Real-time Expert Consultation: GAMKERSGPT")
            st.write("Ask about vulnerabilities, best practices, and threat detection.")
            st.button("💬 Start Chat", key="start_chat")
    st.write("---")
    with st.container():
        l, r = st.columns(2)
        with l:
            st.header("Comprehensive Security Suite")
            c1, c2 = st.columns(2)
            with c1:
                st.write("🔍 Code Obfuscation\n\n🌐 Network Analysis\n\n📂 File Operations")
            with c2:
                st.write("⚠️ Payload Detection\n\n🔐 API Security\n\n🛡️ Anti-Analysis")
        with r:
            st_lottie(animations["security"], height=300, key="security_anim")
    st.write("---")
    c1, c2, c3 = st.columns(3)
    c1.metric("Languages Supported", "7+")
    c2.metric("Security Rules", "1000+")
    c3.metric("Analysis Speed", "<2 min")
    st.write("---")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP CLASS
# ══════════════════════════════════════════════════════════════════════════════

class SecurityAnalysisApp:
    def __init__(self, groq_api_key: str):
        self.chat_model = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="openai/gpt-oss-120b",
            temperature=0.7,
            max_tokens=None,
        )
        self.forensics = BinaryForensics()
        self._build_chains()

    def _build_chains(self):
        # ── Source code analysis prompt ────────────────────────────────────────
        self.analysis_template = PromptTemplate(
            input_variables=["code_chunk"],
            template="""You are a malware analyst. Analyze the following code for malicious indicators.

Code:
{code_chunk}

Return ONLY valid JSON, no extra text:
{{
    "summary": ["key findings"],
    "sections": {{
        "code_obfuscation_techniques": {{"findings": [], "description": ""}},
        "suspicious_api_calls": {{"findings": [], "description": ""}},
        "anti_analysis_mechanisms": {{"findings": [], "description": ""}},
        "network_communication_patterns": {{"findings": [], "description": ""}},
        "file_system_operations": {{"findings": [], "description": ""}},
        "potential_payload_analysis": {{"findings": [], "description": ""}}
    }}
}}
Use "None identified" if nothing found.""",
        )

        # ── Binary / shellcode analysis prompt — much more specific ───────────
        self.binary_analysis_template = PromptTemplate(
            input_variables=["strings_chunk", "forensics_report"],
            template="""You are a senior malware reverse engineer specializing in Metasploit/msfvenom payloads, shellcode, and Windows PE malware.

STATIC FORENSICS PRE-ANALYSIS (already computed):
{forensics_report}

EXTRACTED STRINGS FROM BINARY:
{strings_chunk}

Your job: cross-reference the forensics data with the strings and produce a detailed verdict.

Key indicators to look for:
- Metasploit/msfvenom: EXITFUNC, shikata_ga_nai, meterpreter, staged/stageless markers, ReflectiveDll
- Shellcode stagers: VirtualAlloc, WriteProcessMemory, CreateRemoteThread, NtCreateThreadEx
- C2 communication: hardcoded IPs, URLs, WSASocket, connect, recv, send patterns
- Encoding: high entropy sections, XOR loops, base64 blobs, hex shellcode strings
- Anti-analysis: IsDebuggerPresent, timing checks, VM detection strings
- Persistence: registry Run keys, CreateService, scheduled tasks
- Process injection: OpenProcess, VirtualAllocEx, SetWindowsHookEx
- Privilege escalation: SeDebugPrivilege, token impersonation, named pipe abuse

Threat scoring:
- HIGH: shellcode signatures found, entropy >7.0, stager APIs present, C2 indicators
- MEDIUM: suspicious APIs, encoded blobs, anti-analysis without confirmed shellcode
- LOW: legitimate tool misuse potential only

Return ONLY valid JSON:
{{
    "threat_level": "HIGH|MEDIUM|LOW|CLEAN",
    "verdict": "One sentence verdict",
    "summary": ["finding 1", "finding 2"],
    "sections": {{
        "payload_identification": {{
            "findings": [],
            "description": "Is this msfvenom/Metasploit? Which payload type? Staged or stageless?"
        }},
        "shellcode_indicators": {{
            "findings": [],
            "description": "Shellcode signatures, NOP sleds, encoder stubs, EXITFUNC markers"
        }},
        "command_and_control": {{
            "findings": [],
            "description": "C2 IPs, URLs, ports, protocol patterns"
        }},
        "process_injection_techniques": {{
            "findings": [],
            "description": "Injection APIs and techniques identified"
        }},
        "anti_analysis_mechanisms": {{
            "findings": [],
            "description": "Anti-debug, anti-VM, timing checks"
        }},
        "persistence_mechanisms": {{
            "findings": [],
            "description": "Registry, services, scheduled tasks"
        }},
        "network_indicators": {{
            "findings": [],
            "description": "Network APIs, IPs, domains, ports"
        }},
        "encoding_obfuscation": {{
            "findings": [],
            "description": "Entropy analysis, encoding schemes, packed sections"
        }}
    }}
}}
Use "None identified" only if genuinely absent.""",
        )

        self.analysis_chain = LLMChain(llm=self.chat_model, prompt=self.analysis_template, verbose=True)
        self.binary_analysis_chain = LLMChain(llm=self.chat_model, prompt=self.binary_analysis_template, verbose=True)
        self.chat_memory = ConversationBufferMemory()
        self.conversation = ConversationChain(llm=self.chat_model, memory=self.chat_memory, verbose=True)

    # ── Utilities ──────────────────────────────────────────────────────────────
    def clean_json(self, response: str) -> str:
        start = response.find('{')
        end = response.rfind('}') + 1
        if start != -1 and end:
            response = response[start:end]
        return response.replace('```json', '').replace('```', '').strip()

    def split_chunks(self, content: str, size: int = 12800) -> List[str]:
        return [content[i:i + size] for i in range(0, len(content), size)]

    def _error_analysis(self, error_type: str, details: str, is_binary: bool = False) -> Dict:
        sections = (
            ["payload_identification", "shellcode_indicators", "command_and_control",
             "process_injection_techniques", "anti_analysis_mechanisms",
             "persistence_mechanisms", "network_indicators", "encoding_obfuscation"]
            if is_binary else
            ["code_obfuscation_techniques", "suspicious_api_calls", "anti_analysis_mechanisms",
             "network_communication_patterns", "file_system_operations", "potential_payload_analysis"]
        )
        return {
            "error": f"{error_type}: {details}",
            "threat_level": "UNKNOWN",
            "verdict": "Analysis failed",
            "summary": [f"Analysis failed — {error_type}"],
            "sections": {s: {"findings": ["Analysis failed"], "description": "Technical error"} for s in sections},
        }

    # ── Source code analysis ───────────────────────────────────────────────────
    def analyze_code(self, code_content: str) -> Dict:
        chunks = self.split_chunks(code_content)
        analyses, bar, status = [], st.progress(0), st.empty()
        for i, chunk in enumerate(chunks, 1):
            status.text(f"Analyzing chunk {i}/{len(chunks)}...")
            try:
                resp = self.analysis_chain.predict(code_chunk=chunk)
                cleaned = self.clean_json(resp)
                analyses.append(json.loads(cleaned))
            except Exception as e:
                analyses.append(self._error_analysis("Chunk failed", str(e)))
            bar.progress(i / len(chunks))
        status.text("Analysis complete!")
        bar.empty()
        return self.combine_analyses(analyses)

    # ── Binary analysis ────────────────────────────────────────────────────────
    def extract_strings(self, binary_data: bytes, min_length: int = 6) -> List[str]:
        """Return a list of printable strings from the binary."""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.bin') as tmp:
            tmp.write(binary_data)
            tmp_path = tmp.name
        try:
            try:
                result = subprocess.run(
                    ['strings', '-n', str(min_length), tmp_path],
                    capture_output=True, text=True, check=True,
                )
                lines = result.stdout.splitlines()
            except (subprocess.SubprocessError, FileNotFoundError):
                lines = self._manual_strings(binary_data, min_length)
            return [l for l in lines if re.search(r'[a-zA-Z0-9_]', l)]
        finally:
            try: os.unlink(tmp_path)
            except: pass

    def _manual_strings(self, data: bytes, min_len: int = 6) -> List[str]:
        strings, cur = [], ""
        for b in data:
            if 32 <= b <= 126:
                cur += chr(b)
            else:
                if len(cur) >= min_len:
                    strings.append(cur)
                cur = ""
        if len(cur) >= min_len:
            strings.append(cur)
        return strings

    def analyze_binary(self, binary_data: bytes) -> Dict:
        # Step 1: Static forensics (no LLM needed)
        forensics = BinaryForensics.full_analysis(binary_data)
        string_list = self.extract_strings(binary_data)
        suspicious = BinaryForensics.find_suspicious_strings(string_list)

        # Enrich forensics with string-level findings
        forensics["suspicious_string_findings"] = suspicious

        # Step 2: Build forensics summary to inject into prompt
        forensics_text = self._format_forensics(forensics)

        # Step 3: Chunk the raw strings and send to LLM with forensics context
        strings_blob = "\n".join(string_list)
        chunks = self.split_chunks(strings_blob)
        analyses, bar, status = [], st.progress(0), st.empty()

        for i, chunk in enumerate(chunks, 1):
            status.text(f"Analyzing binary chunk {i}/{len(chunks)}...")
            try:
                resp = self.binary_analysis_chain.predict(
                    strings_chunk=chunk,
                    forensics_report=forensics_text,
                )
                cleaned = self.clean_json(resp)
                analyses.append(json.loads(cleaned))
            except Exception as e:
                analyses.append(self._error_analysis("Chunk failed", str(e), is_binary=True))
            bar.progress(i / len(chunks))

        status.text("Binary analysis complete!")
        bar.empty()

        combined = self.combine_analyses(analyses, is_binary=True)

        # Step 4: Inject static forensics directly into results (no LLM can miss these)
        combined["static_forensics"] = forensics
        combined["static_forensics_summary"] = forensics_text
        return combined

    def _format_forensics(self, f: Dict) -> str:
        lines = [
            f"Overall entropy: {f.get('overall_entropy')} — {f.get('entropy_verdict')}",
            f"PE binary: {f['pe_info'].get('is_pe')}",
            f"Architecture: {f['pe_info'].get('architecture', 'N/A')}",
            f"PE characteristics: {', '.join(f['pe_info'].get('characteristics', []))}",
            f"NOP sled detected: {f.get('nop_sled_detected')} {f.get('nop_sled_offset', '')}",
            f"EXITFUNC markers: {f.get('exitfunc_markers', [])}",
            f"Shellcode signatures: {f.get('shellcode_signatures', [])}",
            f"Suspicious imports: {f.get('suspicious_imports', [])}",
            f"High entropy sections: {f.get('high_entropy_sections', [])}",
        ]
        sf = f.get("suspicious_string_findings", {})
        for k, v in sf.items():
            if v:
                lines.append(f"{k.replace('_',' ').title()}: {v[:5]}")
        return "\n".join(lines)

    # ── Combine multiple chunk analyses ────────────────────────────────────────
    def combine_analyses(self, analyses: List[Dict], is_binary: bool = False) -> Dict:
        if not analyses:
            return {"summary": ["No results"], "sections": {}, "errors": []}
        combined = {"summary": set(), "sections": {}, "errors": [],
                    "threat_level": "CLEAN", "verdict": ""}
        threat_order = {"HIGH": 3, "MEDIUM": 2, "LOW": 1, "CLEAN": 0, "UNKNOWN": 0}
        max_threat = 0

        if "sections" in analyses[0]:
            for s in analyses[0]["sections"]:
                combined["sections"][s] = {"findings": set(), "description": ""}

        for a in analyses:
            if "error" in a:
                combined["errors"].append(a["error"])
            combined["summary"].update(a.get("summary", []))
            tl = a.get("threat_level", "CLEAN")
            if threat_order.get(tl, 0) > max_threat:
                max_threat = threat_order[tl]
                combined["threat_level"] = tl
                combined["verdict"] = a.get("verdict", "")
            for sec, content in a.get("sections", {}).items():
                if sec not in combined["sections"]:
                    combined["sections"][sec] = {"findings": set(), "description": ""}
                combined["sections"][sec]["findings"].update(content.get("findings", []))
                if content.get("description") and not combined["sections"][sec]["description"]:
                    combined["sections"][sec]["description"] = content["description"]

        result = {
            "threat_level": combined["threat_level"],
            "verdict": combined["verdict"],
            "summary": list(combined["summary"]),
            "sections": {},
            "errors": combined["errors"],
        }
        for sec, content in combined["sections"].items():
            findings = list(content["findings"])
            if len(findings) > 1 and "Analysis failed" in findings:
                findings.remove("Analysis failed")
            result["sections"][sec] = {
                "findings": findings,
                "description": content["description"] or "No significant findings.",
            }
        return result

    # ── DOCX report ────────────────────────────────────────────────────────────
    def create_report(self, results: Dict, title: str = "Security Analysis Report") -> str:
        doc = Document()
        doc.add_heading(title, 0)

        # Threat level banner
        tl = results.get("threat_level", "UNKNOWN")
        doc.add_heading(f"Threat Level: {tl}", level=1)
        if results.get("verdict"):
            doc.add_paragraph(results["verdict"], style='Body Text')

        # Static forensics section (binary only)
        sf = results.get("static_forensics")
        if sf:
            doc.add_heading("Static Forensics (Pre-LLM Analysis)", level=1)
            doc.add_paragraph(results.get("static_forensics_summary", ""), style='Body Text')

        doc.add_heading("Executive Summary", level=1)
        for pt in results.get("summary", ["No summary"]):
            doc.add_paragraph(pt, style='Body Text')

        for sec_name, content in results.get("sections", {}).items():
            doc.add_heading(sec_name.replace('_', ' ').title(), level=1)
            if content.get("description"):
                doc.add_paragraph(content["description"], style='Body Text')
            if content.get("findings"):
                doc.add_heading("Findings:", level=2)
                for f in content["findings"]:
                    if f != "None identified":
                        doc.add_paragraph(f"• {f}", style='List Bullet')
                    else:
                        doc.add_paragraph("No issues identified.", style='Body Text')

        if results.get("errors"):
            doc.add_heading("Analysis Errors", level=1)
            for e in results["errors"]:
                doc.add_paragraph(f"• {e}", style='List Bullet')

        fname = f"security_report_{os.getpid()}.docx"
        doc.save(fname)
        return fname

    def get_chat_response(self, user_input: str) -> str:
        return self.conversation.predict(input=user_input + " Be short and crisp.")


# ══════════════════════════════════════════════════════════════════════════════
# DISPLAY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

THREAT_COLORS = {"HIGH": "#ff4444", "MEDIUM": "#ffaa00", "LOW": "#44ff88", "CLEAN": "#44ff88"}

def display_threat_banner(results: Dict):
    tl = results.get("threat_level", "UNKNOWN")
    color = THREAT_COLORS.get(tl, "#9D4EDD")
    verdict = results.get("verdict", "")
    st.markdown(
        f"""<div style='background:#1a0a2e;border-left:6px solid {color};
        border-radius:8px;padding:16px;margin:12px 0;'>
        <h3 style='color:{color};margin:0'>⚠️ Threat Level: {tl}</h3>
        <p style='color:#ccc;margin:4px 0 0'>{verdict}</p></div>""",
        unsafe_allow_html=True,
    )

def display_static_forensics(results: Dict):
    sf = results.get("static_forensics")
    if not sf:
        return
    with st.expander("🔬 Static Forensics (Pre-LLM)", expanded=True):
        col1, col2, col3 = st.columns(3)
        col1.metric("Entropy", sf.get("overall_entropy", "N/A"))
        col2.metric("PE Binary", "✅" if sf["pe_info"].get("is_pe") else "❌")
        col3.metric("Arch", sf["pe_info"].get("architecture", "N/A"))

        if sf.get("shellcode_signatures"):
            st.error("🚨 Shellcode Signatures Found!")
            for sig in sf["shellcode_signatures"]:
                st.code(sig)

        if sf.get("exitfunc_markers"):
            st.error(f"🚨 EXITFUNC markers (msfvenom indicator) at: {sf['exitfunc_markers']}")

        if sf.get("nop_sled_detected"):
            st.warning(f"⚠️ NOP sled detected at {sf.get('nop_sled_offset')}")

        st.write(f"**Entropy verdict:** {sf.get('entropy_verdict')}")

        hi_ent = sf.get("high_entropy_sections", [])
        if hi_ent:
            st.warning(f"⚠️ {len(hi_ent)} high-entropy section(s) — possible packed/encrypted payload:")
            for s in hi_ent[:5]:
                st.code(s)

        sus = sf.get("suspicious_string_findings", {})
        for category, items in sus.items():
            if items:
                st.markdown(f"**{category.replace('_',' ').title()}**")
                for item in items[:8]:
                    st.markdown(f"- `{item}`")

def display_analysis_results(results: Dict):
    display_threat_banner(results)
    display_static_forensics(results)

    st.header("Executive Summary")
    for pt in results.get("summary", ["No summary"]):
        st.write(pt)
    st.divider()

    if results.get("errors"):
        st.error("Analysis Errors")
        for e in results["errors"]:
            st.write(f"• {e}")
        st.divider()

    for sec_name, content in results.get("sections", {}).items():
        st.subheader(sec_name.replace('_', ' ').title())
        if content.get("description"):
            st.write(content["description"])
        if content.get("findings"):
            for finding in content["findings"]:
                st.write(f"• {finding}")
        st.divider()

def download_report_button(fname: str, label: str = "📥 Download Report"):
    with open(fname, "rb") as f:
        st.download_button(label=label, data=f, file_name=fname,
                           mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    os.remove(fname)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    groq_api_key = api_key_sidebar()

    with st.sidebar:
        st.write("---")
        tabs = on_hover_tabs(
            tabName=['Code Analyzer', 'GAMKERSGPT', 'About'],
            iconName=['code', 'chat', 'info'],
            styles={
                'navtab': {'background-color': 'black', 'color': '#9D4EDD',
                           'font-size': '16px', 'transition': '.3s',
                           'white-space': 'nowrap', 'text-transform': 'uppercase'},
                'tabOptionsStyle': {':hover': {'color': '#1A1A1A', 'background-color': 'black'}},
                'iconStyle': {'position': 'fixed', 'left': '7.5px', 'text-align': 'left'},
                'tabStyle': {'list-style-type': 'none', 'margin-bottom': '30px', 'padding-left': '30px'},
            },
        )

    if tabs == 'About':
        about_section()
        return

    if not groq_api_key:
        st.info("👈 Enter your Groq API key in the sidebar to get started.")
        return

    if "app" not in st.session_state or st.session_state.get("_api_key") != groq_api_key:
        st.session_state.app = SecurityAnalysisApp(groq_api_key)
        st.session_state.messages = []
        st.session_state._api_key = groq_api_key

    # ── Code Analyzer ──────────────────────────────────────────────────────────
    if tabs == 'Code Analyzer':
        st.title("GAMKERS Security Analyzer")
        tab1, tab2, tab3 = st.tabs(["📝 Paste Code", "📁 Upload Source File", "💾 Upload Binary"])

        with tab1:
            code_input = st.text_area("Paste your code here:", height=300)
            if st.button("🔍 Analyze Code", key="analyze_pasted") and code_input:
                with st.spinner("🔄 Analyzing..."):
                    results = st.session_state.app.analyze_code(code_input)
                    display_analysis_results(results)
                    download_report_button(st.session_state.app.create_report(results))

        with tab2:
            uploaded_file = st.file_uploader("Choose a source code file",
                type=['py', 'js', 'java', 'cpp', 'cs', 'php', 'rb'], key="code_file")
            if st.button("🔍 Analyze Source File", key="analyze_source") and uploaded_file:
                with st.spinner("🔄 Analyzing..."):
                    results = st.session_state.app.analyze_code(uploaded_file.read().decode())
                    display_analysis_results(results)
                    download_report_button(st.session_state.app.create_report(results))

        with tab3:
            st.write("Upload a binary file for deep static + AI analysis")
            uploaded_binary = st.file_uploader("Choose a binary file", type=['exe', 'bin', 'dll'], key="binary_file")
            if uploaded_binary:
                st.info("Runs entropy analysis, PE inspection, shellcode signature matching, and AI string analysis.")
            if st.button("🔍 Analyze Binary", key="analyze_binary") and uploaded_binary:
                with st.spinner("🔄 Running static forensics + AI analysis..."):
                    binary_data = uploaded_binary.read()

                    with st.expander("📄 Extracted Strings Preview"):
                        preview_strings = st.session_state.app.extract_strings(binary_data)
                        st.text_area("Strings", value="\n".join(preview_strings)[:10000] +
                                     ("\n\n[Truncated...]" if len(preview_strings) > 200 else ""),
                                     height=250, disabled=True)

                    results = st.session_state.app.analyze_binary(binary_data)
                    st.subheader("Binary Analysis Results")
                    display_analysis_results(results)
                    download_report_button(
                        st.session_state.app.create_report(
                            results, title=f"Binary Analysis — {uploaded_binary.name}"
                        ),
                        label="📥 Download Binary Report",
                    )

    # ── GAMKERSGPT ─────────────────────────────────────────────────────────────
    elif tabs == 'GAMKERSGPT':
        st.title("GAMKERSGPT - Security Assistant")
        st.markdown("""
            <style>
            [data-testid="stChatMessage"]{background-color:#000!important;color:#fff!important;padding:10px;border-radius:8px;}
            [data-testid="stChatInput"]{background-color:#000!important;color:#fff!important;border:1px solid #333;border-radius:8px;}
            [data-testid="stBottom"]{background-color:#000!important;padding:10px!important;border-radius:8px!important;}
            </style>""", unsafe_allow_html=True)
        st.markdown("### Welcome to GAMKERSGPT!")
        st.write("Ask about vulnerabilities, best practices, threat analysis, and more.")
        st.info("Tip: Be as specific as possible.")

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("Ask your cyber security question..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("🤔 Thinking..."):
                    response = st.session_state.app.get_chat_response(prompt)
                    response = response.split("</think>")[-1].strip()
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
