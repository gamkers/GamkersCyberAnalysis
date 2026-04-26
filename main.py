import streamlit as st
from st_on_hover_tabs import on_hover_tabs
from typing import List, Dict, Tuple, Optional
from langchain_groq import ChatGroq
from langchain_classic.chains import LLMChain, ConversationChain
from langchain_classic.prompts import PromptTemplate
from langchain_classic.memory import ConversationBufferMemory
import json, math, struct, re, os, subprocess, tempfile
from collections import Counter
from docx import Document
from streamlit_lottie import st_lottie

# ── Streamlit chrome ───────────────────────────────────────────────────────────
st.markdown("<style>MainMenu{visibility:hidden;}footer{visibility:hidden;}</style>", unsafe_allow_html=True)
try:
    with open("style.css") as f:
        st.markdown('<style>' + f.read() + '</style>', unsafe_allow_html=True)
except Exception:
    pass

dark_purple_theme = """<style>
:root{--primary-color:#7B2CBF;--background-color:black;--secondary-bg:#2D2D2D;--text-color:#FFFFFF;--accent-color:#9D4EDD;}
.stApp{background-color:var(--background-color);color:var(--text-color);}
h1,h2,h3{color:var(--accent-color)!important;}
.stButton>button{background-color:var(--primary-color);color:white;border:none;border-radius:4px;transition:all 0.3s ease;}
.stButton>button:hover{background-color:var(--accent-color);transform:translateY(-2px);}
.stTextArea>div>div>textarea{background-color:var(--secondary-bg);color:var(--text-color);border:1px solid var(--primary-color);}
.stFileUploader{background-color:var(--secondary-bg);border:1px dashed var(--primary-color);border-radius:4px;}
.stProgress>div>div>div>div{background-color:var(--primary-color);}
.stChatMessage{background-color:var(--secondary-bg);border-radius:8px;padding:10px;margin:5px 0;}
</style>"""
st.markdown(dark_purple_theme, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PE PARSER
# ══════════════════════════════════════════════════════════════════════════════

class PEParser:
    """
    Parses the full PE (Portable Executable) structure from raw bytes.
    Covers: DOS header, PE/COFF header, Optional header, section table,
    import table (IAT), and export table.
    """

    MACHINE_TYPES = {
        0x0000: "Unknown", 0x014C: "x86 (32-bit)",
        0x0200: "IA-64",   0x8664: "x86-64 (64-bit)",
        0x01C4: "ARMv7",   0xAA64: "ARM64",
    }

    SECTION_FLAGS = {
        0x00000020: "CODE",        0x00000040: "INITIALIZED_DATA",
        0x00000080: "UNINITIALIZED_DATA",
        0x20000000: "EXECUTE",     0x40000000: "READ",
        0x80000000: "WRITE",
    }

    CHARACTERISTICS = {
        0x0001: "NO_RELOCS",       0x0002: "EXECUTABLE",
        0x0004: "NO_LINE_NUMS",    0x0020: "LARGE_ADDR_AWARE",
        0x0100: "32BIT_MACHINE",   0x1000: "SYSTEM",
        0x2000: "DLL",             0x4000: "UP_SYSTEM_ONLY",
    }

    DLL_CHARACTERISTICS = {
        0x0020: "HIGH_ENTROPY_VA", 0x0040: "DYNAMIC_BASE (ASLR)",
        0x0080: "FORCE_INTEGRITY", 0x0100: "NX_COMPAT (DEP)",
        0x0400: "NO_ISOLATION",    0x0800: "NO_SEH",
        0x1000: "NO_BIND",         0x4000: "WDM_DRIVER",
        0x8000: "TERMINAL_SERVER",
    }

    def __init__(self, data: bytes):
        self.data = data
        self.valid = False
        self.is_64bit = False
        self.pe_offset = 0
        self.sections: List[Dict] = []
        self.imports: Dict[str, List[str]] = {}
        self.exports: List[str] = []
        self.result: Dict = {}
        self._parse()

    def _u8(self, off): return struct.unpack_from("B", self.data, off)[0]
    def _u16(self, off): return struct.unpack_from("<H", self.data, off)[0]
    def _u32(self, off): return struct.unpack_from("<I", self.data, off)[0]
    def _u64(self, off): return struct.unpack_from("<Q", self.data, off)[0]

    def _rva_to_offset(self, rva: int) -> Optional[int]:
        for s in self.sections:
            va = s["virtual_address"]
            size = s["virtual_size"] or s["raw_size"]
            if va <= rva < va + size:
                return s["raw_offset"] + (rva - va)
        return None

    def _read_string(self, offset: int, max_len: int = 256) -> str:
        end = self.data.find(b'\x00', offset, offset + max_len)
        if end == -1:
            end = offset + max_len
        return self.data[offset:end].decode('latin-1', errors='replace')

    def _parse(self):
        d = self.data
        r = {}
        try:
            # ── DOS Header ────────────────────────────────────────────────────
            if d[:2] != b'MZ':
                self.result = {"error": "Not a PE file (missing MZ signature)"}
                return
            self.pe_offset = self._u32(0x3C)
            pe = self.pe_offset

            if d[pe:pe+4] != b'PE\x00\x00':
                self.result = {"error": "Missing PE signature"}
                return
            self.valid = True

            # ── COFF Header ───────────────────────────────────────────────────
            machine    = self._u16(pe + 4)
            num_sects  = self._u16(pe + 6)
            timestamp  = self._u32(pe + 8)
            opt_size   = self._u16(pe + 20)
            chars      = self._u16(pe + 22)

            r["architecture"]    = self.MACHINE_TYPES.get(machine, f"0x{machine:04X}")
            r["num_sections"]    = num_sects
            r["timestamp"]       = timestamp
            r["timestamp_human"] = self._ts(timestamp)
            r["characteristics"] = [v for k, v in self.CHARACTERISTICS.items() if chars & k]

            # ── Optional Header ───────────────────────────────────────────────
            opt = pe + 24
            magic = self._u16(opt)
            self.is_64bit = (magic == 0x20B)   # PE32+ = 64-bit
            r["is_64bit"] = self.is_64bit
            r["pe_type"]  = "PE32+" if self.is_64bit else "PE32"

            r["entry_point"]     = f"0x{self._u32(opt+16):08X}"
            r["image_base"]      = f"0x{(self._u64(opt+24) if self.is_64bit else self._u32(opt+28)):016X}"
            r["size_of_image"]   = f"0x{self._u32(opt+56):08X}"
            r["size_of_headers"] = f"0x{self._u32(opt+60):08X}"
            r["subsystem"]       = {1:"Native", 2:"Windows GUI", 3:"Windows CUI",
                                     9:"WinCE GUI", 14:"EFI", 16:"EFI Runtime"}.get(self._u16(opt+68), "Unknown")
            dll_chars            = self._u16(opt+70)
            r["dll_characteristics"] = [v for k, v in self.DLL_CHARACTERISTICS.items() if dll_chars & k]

            # Pointer to data directories
            dd_offset = opt + (112 if self.is_64bit else 96)

            # ── Section Table ─────────────────────────────────────────────────
            sect_offset = opt + opt_size
            sections = []
            for i in range(num_sects):
                base = sect_offset + i * 40
                name = d[base:base+8].rstrip(b'\x00').decode('latin-1', errors='replace')
                vsz  = self._u32(base + 8)
                va   = self._u32(base + 12)
                rsz  = self._u32(base + 16)
                roff = self._u32(base + 20)
                sc   = self._u32(base + 36)
                flags = [v for k, v in self.SECTION_FLAGS.items() if sc & k]
                raw   = d[roff:roff + rsz] if roff and rsz else b''
                ent   = self._entropy(raw)
                sections.append({
                    "name": name, "virtual_address": va, "virtual_size": vsz,
                    "raw_offset": roff, "raw_size": rsz,
                    "flags": flags, "entropy": round(ent, 3),
                    "suspicious": ent > 7.0 or ("EXECUTE" in flags and "WRITE" in flags),
                })
            self.sections = sections
            r["sections"] = sections

            # ── Import Table ──────────────────────────────────────────────────
            imp_rva = self._u32(dd_offset)       # DataDirectory[1] = imports
            if imp_rva:
                r["imports"] = self._parse_imports(imp_rva)
                self.imports = r["imports"]

            # ── Export Table ──────────────────────────────────────────────────
            exp_rva = self._u32(dd_offset + 0)   # DataDirectory[0] = exports (index 0)
            # Actually exports are DataDirectory[0]:
            exp_rva_real = self._u32(dd_offset - 8) if dd_offset >= 8 else 0
            r["exports"] = self._parse_exports(exp_rva_real) if exp_rva_real else []
            self.exports = r["exports"]

        except Exception as ex:
            r["parse_error"] = str(ex)

        self.result = r

    def _parse_imports(self, imp_rva: int) -> Dict[str, List[str]]:
        imports = {}
        try:
            off = self._rva_to_offset(imp_rva)
            if off is None:
                return imports
            while True:
                # IMAGE_IMPORT_DESCRIPTOR is 20 bytes
                ilt_rva  = self._u32(off)
                name_rva = self._u32(off + 12)
                iat_rva  = self._u32(off + 16)
                if name_rva == 0:
                    break
                name_off = self._rva_to_offset(name_rva)
                dll_name = self._read_string(name_off) if name_off else "Unknown"
                funcs = []
                lookup_rva = ilt_rva or iat_rva
                if lookup_rva:
                    loff = self._rva_to_offset(lookup_rva)
                    if loff:
                        while True:
                            entry = self._u64(loff) if self.is_64bit else self._u32(loff)
                            loff += 8 if self.is_64bit else 4
                            if entry == 0:
                                break
                            ordinal_flag = (1 << 63) if self.is_64bit else (1 << 31)
                            if entry & ordinal_flag:
                                funcs.append(f"Ordinal#{entry & 0xFFFF}")
                            else:
                                fn_off = self._rva_to_offset(entry & 0x7FFFFFFF)
                                if fn_off:
                                    funcs.append(self._read_string(fn_off + 2))
                imports[dll_name] = funcs
                off += 20
        except Exception:
            pass
        return imports

    def _parse_exports(self, exp_rva: int) -> List[str]:
        exports = []
        try:
            off = self._rva_to_offset(exp_rva)
            if off is None:
                return exports
            num_names = self._u32(off + 24)
            names_rva = self._u32(off + 32)
            names_off = self._rva_to_offset(names_rva)
            if names_off is None:
                return exports
            for i in range(min(num_names, 500)):
                n_rva = self._u32(names_off + i * 4)
                n_off = self._rva_to_offset(n_rva)
                if n_off:
                    exports.append(self._read_string(n_off))
        except Exception:
            pass
        return exports

    @staticmethod
    def _entropy(data: bytes) -> float:
        if not data:
            return 0.0
        c = Counter(data)
        l = len(data)
        return -sum((v / l) * math.log2(v / l) for v in c.values())

    @staticmethod
    def _ts(ts: int) -> str:
        import datetime
        try:
            return datetime.datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S UTC')
        except Exception:
            return "Invalid"


# ══════════════════════════════════════════════════════════════════════════════
#  CODE INJECTION DETECTOR
# ══════════════════════════════════════════════════════════════════════════════

class InjectionDetector:
    """
    Detects code injection technique artifacts from PE import tables and strings.
    Covers: process hollowing, DLL injection, reflective DLL, shellcode injection,
    atom bombing, section injection, APC injection, SetWindowsHookEx, IAT hooking.
    """

    # Each technique: { name, required_apis (any match triggers), description }
    TECHNIQUES = [
        {
            "name": "Process Hollowing",
            "apis": ["NtUnmapViewOfSection", "ZwUnmapViewOfSection",
                     "SetThreadContext", "GetThreadContext", "ResumeThread"],
            "also_needs": ["VirtualAllocEx", "WriteProcessMemory"],
            "description": "Hollows a legitimate process and injects malicious PE image",
            "severity": "HIGH",
        },
        {
            "name": "Classic DLL Injection",
            "apis": ["LoadLibraryA", "LoadLibraryW"],
            "also_needs": ["CreateRemoteThread", "VirtualAllocEx", "WriteProcessMemory"],
            "description": "Forces target process to load attacker-controlled DLL via CreateRemoteThread",
            "severity": "HIGH",
        },
        {
            "name": "Reflective DLL Injection",
            "apis": ["ReflectiveLoader", "ReflectiveDllInjection"],
            "also_needs": [],
            "description": "DLL loads itself from memory; evades LoadLibrary-based detection",
            "severity": "HIGH",
        },
        {
            "name": "Shellcode / PE Injection",
            "apis": ["VirtualAllocEx", "WriteProcessMemory"],
            "also_needs": ["CreateRemoteThread", "NtCreateThreadEx", "RtlCreateUserThread"],
            "description": "Allocates memory in remote process and writes+executes shellcode or PE",
            "severity": "HIGH",
        },
        {
            "name": "APC Injection",
            "apis": ["QueueUserAPC", "NtQueueApcThread", "NtQueueApcThreadEx"],
            "also_needs": [],
            "description": "Queues APC to hijack thread execution in target process",
            "severity": "HIGH",
        },
        {
            "name": "Atom Bombing",
            "apis": ["GlobalAddAtomA", "GlobalAddAtomW", "GlobalGetAtomNameA"],
            "also_needs": ["NtQueueApcThread"],
            "description": "Uses Windows atom tables to smuggle shellcode into target process",
            "severity": "HIGH",
        },
        {
            "name": "Section / Shared Memory Injection",
            "apis": ["NtCreateSection", "ZwCreateSection", "NtMapViewOfSection",
                     "ZwMapViewOfSection"],
            "also_needs": [],
            "description": "Creates shared memory section mapped into both attacker and target process",
            "severity": "HIGH",
        },
        {
            "name": "SetWindowsHookEx Injection",
            "apis": ["SetWindowsHookExA", "SetWindowsHookExW"],
            "also_needs": [],
            "description": "Installs message hook to force DLL load into target processes",
            "severity": "MEDIUM",
        },
        {
            "name": "IAT Hooking",
            "apis": ["VirtualProtect"],
            "also_needs": ["GetProcAddress", "GetModuleHandleA", "GetModuleHandleW"],
            "description": "Modifies Import Address Table to redirect API calls",
            "severity": "MEDIUM",
        },
        {
            "name": "Early Bird APC",
            "apis": ["CreateProcessA", "CreateProcessW"],
            "also_needs": ["VirtualAllocEx", "WriteProcessMemory", "QueueUserAPC"],
            "description": "Injects APC before process main thread initializes",
            "severity": "HIGH",
        },
        {
            "name": "Process Doppelgänging",
            "apis": ["NtCreateTransaction", "NtWriteFile", "NtCreateSection",
                     "NtRollbackTransaction"],
            "also_needs": [],
            "description": "Uses NTFS transactions to create process from phantom file on disk",
            "severity": "HIGH",
        },
        {
            "name": "Thread Execution Hijacking",
            "apis": ["SuspendThread", "SetThreadContext", "ResumeThread"],
            "also_needs": ["VirtualAllocEx", "WriteProcessMemory"],
            "description": "Suspends existing thread, redirects EIP/RIP to shellcode",
            "severity": "HIGH",
        },
    ]

    # Privilege escalation / evasion APIs
    EVASION_APIS = {
        "anti_debug": ["IsDebuggerPresent", "CheckRemoteDebuggerPresent",
                       "NtQueryInformationProcess", "OutputDebugStringA",
                       "FindWindowA", "FindWindowW"],
        "anti_vm":    ["GetSystemInfo", "cpuid", "vpcext", "vmware",
                       "VirtualBox", "VBOX", "QEMU"],
        "timing":     ["GetTickCount", "GetTickCount64", "QueryPerformanceCounter",
                       "timeGetTime", "NtDelayExecution", "Sleep"],
        "privilege":  ["AdjustTokenPrivileges", "LookupPrivilegeValueA",
                       "SeDebugPrivilege", "OpenProcessToken",
                       "ImpersonateLoggedOnUser", "DuplicateTokenEx"],
        "persistence":["RegSetValueExA", "RegSetValueExW", "RegCreateKeyExA",
                       "CreateServiceA", "CreateServiceW", "OpenSCManagerA",
                       "SchRpcRegisterTask", "ITaskScheduler"],
    }

    @classmethod
    def detect(cls, imports: Dict[str, List[str]], string_list: List[str]) -> Dict:
        """Run all injection technique detectors against import table + strings."""
        all_apis = set()
        for dll_funcs in imports.values():
            all_apis.update(f.lower() for f in dll_funcs)
        # Also check raw strings (handles runtime GetProcAddress resolution)
        all_apis.update(s.lower() for s in string_list if len(s) < 60)

        results = {
            "detected_techniques": [],
            "evasion_capabilities": {},
            "suspicious_api_combinations": [],
            "risk_score": 0,
        }

        for technique in cls.TECHNIQUES:
            primary_hit = [a for a in technique["apis"] if a.lower() in all_apis]
            secondary_hit = [a for a in technique["also_needs"] if a.lower() in all_apis]

            # Trigger if any primary API matches
            if primary_hit:
                confidence = "HIGH" if (not technique["also_needs"] or secondary_hit) else "MEDIUM"
                results["detected_techniques"].append({
                    "technique": technique["name"],
                    "severity": technique["severity"],
                    "confidence": confidence,
                    "description": technique["description"],
                    "matched_apis": primary_hit + secondary_hit,
                })
                results["risk_score"] += 3 if confidence == "HIGH" else 1

        # Evasion detection
        for category, apis in cls.EVASION_APIS.items():
            hits = [a for a in apis if a.lower() in all_apis]
            if hits:
                results["evasion_capabilities"][category] = hits
                results["risk_score"] += 1

        # Flag suspicious API combos even without confirmed technique
        inject_alloc  = "virtualallocex" in all_apis
        inject_write  = "writeprocessmemory" in all_apis
        inject_thread = any(a in all_apis for a in ["createremotethread", "ntcreatethreadex",
                                                     "rtlcreateuserthread"])
        if inject_alloc and inject_write and inject_thread:
            results["suspicious_api_combinations"].append(
                "Classic injection triad: VirtualAllocEx + WriteProcessMemory + CreateRemoteThread"
            )
            results["risk_score"] += 5

        return results


# ══════════════════════════════════════════════════════════════════════════════
#  BINARY FORENSICS (entropy + shellcode sigs)
# ══════════════════════════════════════════════════════════════════════════════

class BinaryForensics:
    SHELLCODE_SIGS = [
        (b"\xfc\xe8\x82\x00\x00\x00", "Meterpreter x86 prologue"),
        (b"\xfc\xe8\x89\x00\x00\x00", "Meterpreter x86 variant"),
        (b"\xfc\x48\x83\xe4\xf0\xe8", "Meterpreter x64 prologue"),
        (b"\xd9\xeb\x9b\xd9\x74\x24", "Shikata_ga_nai encoder stub"),
        (b"\x31\xc9\x83\xe9",         "XOR decode loop"),
        (b"\x60\x89\xe5\x31\xd2",     "Egghunter pattern"),
        (b"\xeb\x27\x5e\x89\x76",     "Jump-call-pop shellcode"),
        (b"\x4d\x5a\x90\x00\x03\x00", "Embedded MZ/PE header"),
    ]

    ENCODER_ARTIFACTS = [
        "shikata", "EXITFUNC", "metsrv", "meterpreter", "ReflectiveDll",
        "reverse_tcp", "reverse_https", "bind_tcp", "migrate",
        "getsystem", "hashdump", "staged", "stager", "msf",
    ]

    @staticmethod
    def shannon_entropy(data: bytes) -> float:
        if not data: return 0.0
        c = Counter(data)
        l = len(data)
        return -sum((v / l) * math.log2(v / l) for v in c.values())

    @classmethod
    def full_analysis(cls, data: bytes, pe: PEParser) -> Dict:
        r = {}
        r["overall_entropy"]  = round(cls.shannon_entropy(data), 3)
        r["entropy_verdict"]  = (
            "🔴 HIGH — packed/encrypted/shellcode likely" if r["overall_entropy"] > 7.0 else
            "🟡 MEDIUM — possible compression or encoding"  if r["overall_entropy"] > 6.0 else
            "🟢 LOW — mostly plaintext content"
        )

        # Shellcode signatures in raw bytes
        r["shellcode_signatures"] = []
        for sig, label in cls.SHELLCODE_SIGS:
            off = data.find(sig)
            if off != -1:
                r["shellcode_signatures"].append(
                    f"{label} @ 0x{off:08X}: {sig.hex()}"
                )

        # EXITFUNC markers (msfvenom-specific)
        r["exitfunc_markers"] = [f"0x{m.start():08X}" for m in re.finditer(b"EXITFUNC", data)]

        # NOP sled
        nop = data.find(b"\x90" * 16)
        r["nop_sled"] = f"0x{nop:08X}" if nop != -1 else None

        # Encoder artifact strings
        r["encoder_artifacts"] = [
            a for a in cls.ENCODER_ARTIFACTS if a.encode('latin-1') in data
        ]

        # Suspicious sections from PE parser
        r["suspicious_sections"] = [
            f"{s['name']}: entropy={s['entropy']}, flags={s['flags']}"
            for s in pe.sections if s.get("suspicious")
        ]

        # Embedded PE check (MZ inside the file past offset 0)
        embedded_mz = [f"0x{m.start():08X}" for m in re.finditer(b"MZ", data[0x200:])]
        r["embedded_pe_headers"] = embedded_mz[:5]

        return r


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

def api_key_sidebar() -> str:
    with st.sidebar:
        st.markdown("### 🔑 Groq API Key")
        key = st.text_input("Enter your Groq API key", type="password",
                            placeholder="gsk_...",
                            help="Never stored — lives only in this browser session.")
        if key:
            st.success("Key loaded ✓", icon="✅")
        else:
            st.warning("Paste your Groq API key to get started.")
    return key


# ══════════════════════════════════════════════════════════════════════════════
#  ABOUT
# ══════════════════════════════════════════════════════════════════════════════

def load_lottie_urls():
    return {
        "analysis": "https://assets8.lottiefiles.com/packages/lf20_qmfs6c3i.json",
        "chat":     "https://assets8.lottiefiles.com/packages/lf20_2LdLki.json",
        "security": "https://assets8.lottiefiles.com/packages/lf20_oyi9a28g.json",
    }

def about_section():
    anim = load_lottie_urls()
    st.title("GAMKERS Security Analysis Suite")
    st.write("---")
    with st.container():
        l, r = st.columns(2)
        with l:
            st.header("Advanced Security Analysis")
            st.write("Deep PE inspection, injection detection, entropy analysis, and AI-powered verdict.")
            st.button("🚀 Try Analysis Now", key="try_analysis")
        with r:
            st_lottie(anim["analysis"], height=300, key="analysis_anim")
    st.write("---")
    with st.container():
        l, r = st.columns(2)
        with l:
            st_lottie(anim["chat"], height=300, key="chat_anim")
        with r:
            st.header("GAMKERSGPT")
            st.write("Expert cybersecurity assistant.")
            st.button("💬 Start Chat", key="start_chat")
    st.write("---")
    c1, c2, c3 = st.columns(3)
    c1.metric("Languages", "7+")
    c2.metric("Injection Techniques", "12")
    c3.metric("Detection Rules", "1000+")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN APP
# ══════════════════════════════════════════════════════════════════════════════

class SecurityAnalysisApp:
    def __init__(self, groq_api_key: str):
        self.chat_model = ChatGroq(groq_api_key=groq_api_key,
                                   model_name="qwen-qwq-32b",
                                   temperature=0.7, max_tokens=None)
        self._build_chains()

    def _build_chains(self):
        self.code_prompt = PromptTemplate(
            input_variables=["code_chunk"],
            template="""You are a malware analyst. Analyze this code for malicious indicators.

Code:
{code_chunk}

Return ONLY valid JSON:
{{
    "summary": ["findings"],
    "sections": {{
        "code_obfuscation_techniques": {{"findings": [], "description": ""}},
        "suspicious_api_calls":        {{"findings": [], "description": ""}},
        "anti_analysis_mechanisms":    {{"findings": [], "description": ""}},
        "network_communication_patterns": {{"findings": [], "description": ""}},
        "file_system_operations":      {{"findings": [], "description": ""}},
        "potential_payload_analysis":  {{"findings": [], "description": ""}}
    }}
}}
Use "None identified" if nothing found.""",
        )

        self.binary_prompt = PromptTemplate(
            input_variables=["strings_chunk", "forensics_report", "pe_report", "injection_report"],
            template="""You are a senior malware reverse engineer specializing in PE malware, code injection, and Metasploit payloads.

═══ STATIC PE ANALYSIS ═══
{pe_report}

═══ BINARY FORENSICS ═══
{forensics_report}

═══ INJECTION TECHNIQUE DETECTION ═══
{injection_report}

═══ EXTRACTED STRINGS ═══
{strings_chunk}

Cross-reference all four sources and produce your verdict. Consider:
- PE sections with WRITE+EXECUTE flags (common in shellcode loaders)
- High-entropy sections (packed/encrypted payload)
- Detected injection techniques and their confidence level
- Shellcode signatures, EXITFUNC markers, NOP sleds
- C2 indicators: hardcoded IPs, URLs, suspicious ports
- Anti-analysis: debugger/VM checks, timing tricks
- Persistence: registry, services, scheduled tasks

Return ONLY valid JSON:
{{
    "threat_level": "HIGH|MEDIUM|LOW|CLEAN",
    "verdict": "One clear sentence verdict naming the malware family/technique if identifiable",
    "summary": ["key finding 1", "key finding 2"],
    "sections": {{
        "payload_identification": {{
            "findings": [],
            "description": "Malware family, payload type, staged vs stageless"
        }},
        "injection_techniques": {{
            "findings": [],
            "description": "Detected injection methods with confidence"
        }},
        "shellcode_indicators": {{
            "findings": [],
            "description": "Shellcode signatures, NOP sleds, encoder stubs, EXITFUNC"
        }},
        "command_and_control": {{
            "findings": [],
            "description": "C2 IPs, URLs, ports, protocol"
        }},
        "anti_analysis": {{
            "findings": [],
            "description": "Anti-debug, anti-VM, timing evasion"
        }},
        "persistence": {{
            "findings": [],
            "description": "Registry, services, scheduled tasks"
        }},
        "encoding_obfuscation": {{
            "findings": [],
            "description": "Encoding schemes, high-entropy sections, packing"
        }},
        "privilege_escalation": {{
            "findings": [],
            "description": "Token manipulation, privilege abuse"
        }}
    }}
}}""",
        )

        self.code_chain   = LLMChain(llm=self.chat_model, prompt=self.code_prompt,   verbose=True)
        self.binary_chain = LLMChain(llm=self.chat_model, prompt=self.binary_prompt, verbose=True)
        self.chat_memory  = ConversationBufferMemory()
        self.conversation = ConversationChain(llm=self.chat_model, memory=self.chat_memory, verbose=True)

    # ── utilities ──────────────────────────────────────────────────────────────
    def _clean_json(self, s: str) -> str:
        start = s.find('{'); end = s.rfind('}') + 1
        if start != -1 and end:
            s = s[start:end]
        return s.replace('```json','').replace('```','').strip()

    def _chunks(self, text: str, size: int = 12800) -> List[str]:
        return [text[i:i+size] for i in range(0, len(text), size)]

    def _error(self, msg: str, binary: bool = False) -> Dict:
        secs = (["payload_identification","injection_techniques","shellcode_indicators",
                 "command_and_control","anti_analysis","persistence",
                 "encoding_obfuscation","privilege_escalation"]
                if binary else
                ["code_obfuscation_techniques","suspicious_api_calls","anti_analysis_mechanisms",
                 "network_communication_patterns","file_system_operations","potential_payload_analysis"])
        return {"threat_level":"UNKNOWN","verdict":"Analysis failed","summary":[msg],
                "sections":{s:{"findings":["Analysis failed"],"description":"Error"} for s in secs},
                "errors":[msg]}

    def _combine(self, analyses: List[Dict]) -> Dict:
        if not analyses:
            return {"summary":[],"sections":{},"errors":[]}
        order = {"HIGH":3,"MEDIUM":2,"LOW":1,"CLEAN":0,"UNKNOWN":0}
        combined = {"summary":set(),"sections":{},"errors":[],
                    "threat_level":"CLEAN","verdict":"","risk_score":0}
        for s in analyses[0].get("sections",{}):
            combined["sections"][s] = {"findings":set(),"description":""}
        for a in analyses:
            if "error" in a: combined["errors"].append(a["error"])
            combined["summary"].update(a.get("summary",[]))
            if order.get(a.get("threat_level","CLEAN"),0) > order.get(combined["threat_level"],0):
                combined["threat_level"] = a.get("threat_level","CLEAN")
                combined["verdict"]      = a.get("verdict","")
            for sec, c in a.get("sections",{}).items():
                if sec not in combined["sections"]:
                    combined["sections"][sec] = {"findings":set(),"description":""}
                combined["sections"][sec]["findings"].update(c.get("findings",[]))
                if c.get("description") and not combined["sections"][sec]["description"]:
                    combined["sections"][sec]["description"] = c["description"]
        result = {k: combined[k] for k in ["threat_level","verdict","errors"]}
        result["summary"] = list(combined["summary"])
        result["sections"] = {}
        for sec, c in combined["sections"].items():
            findings = list(c["findings"])
            if len(findings) > 1 and "Analysis failed" in findings:
                findings.remove("Analysis failed")
            result["sections"][sec] = {"findings": findings,
                                        "description": c["description"] or "No significant findings."}
        return result

    # ── source code ────────────────────────────────────────────────────────────
    def analyze_code(self, code: str) -> Dict:
        chunks = self._chunks(code)
        analyses, bar, status = [], st.progress(0), st.empty()
        for i, chunk in enumerate(chunks, 1):
            status.text(f"Analyzing chunk {i}/{len(chunks)}...")
            try:
                raw = self.code_chain.predict(code_chunk=chunk)
                analyses.append(json.loads(self._clean_json(raw)))
            except Exception as e:
                analyses.append(self._error(str(e)))
            bar.progress(i / len(chunks))
        status.text("Done!"); bar.empty()
        return self._combine(analyses)

    # ── binary ─────────────────────────────────────────────────────────────────
    def extract_strings(self, data: bytes, min_len: int = 6) -> List[str]:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.bin') as tmp:
            tmp.write(data); path = tmp.name
        try:
            try:
                out = subprocess.run(['strings', '-n', str(min_len), path],
                                     capture_output=True, text=True, check=True).stdout
                lines = out.splitlines()
            except Exception:
                lines = self._manual_strings(data, min_len)
            return [l for l in lines if re.search(r'[a-zA-Z0-9_]', l)]
        finally:
            try: os.unlink(path)
            except: pass

    def _manual_strings(self, data: bytes, min_len: int) -> List[str]:
        out, cur = [], ""
        for b in data:
            if 32 <= b <= 126: cur += chr(b)
            else:
                if len(cur) >= min_len: out.append(cur)
                cur = ""
        if len(cur) >= min_len: out.append(cur)
        return out

    def analyze_binary(self, data: bytes) -> Dict:
        status_area = st.empty()

        status_area.info("🔬 Step 1/4 — Parsing PE structure...")
        pe = PEParser(data)

        status_area.info("📊 Step 2/4 — Running entropy & shellcode analysis...")
        forensics = BinaryForensics.full_analysis(data, pe)

        status_area.info("🎯 Step 3/4 — Detecting injection techniques...")
        string_list = self.extract_strings(data)
        injection   = InjectionDetector.detect(pe.imports, string_list)

        # Format static reports for LLM
        pe_text        = self._format_pe(pe)
        forensics_text = self._format_forensics(forensics)
        injection_text = self._format_injection(injection)

        status_area.info("🤖 Step 4/4 — AI analysis...")
        strings_blob = "\n".join(string_list)
        chunks = self._chunks(strings_blob)
        analyses, bar, progress_status = [], st.progress(0), st.empty()
        for i, chunk in enumerate(chunks, 1):
            progress_status.text(f"AI analyzing chunk {i}/{len(chunks)}...")
            try:
                raw = self.binary_chain.predict(
                    strings_chunk=chunk,
                    forensics_report=forensics_text,
                    pe_report=pe_text,
                    injection_report=injection_text,
                )
                analyses.append(json.loads(self._clean_json(raw)))
            except Exception as e:
                analyses.append(self._error(str(e), binary=True))
            bar.progress(i / len(chunks))

        progress_status.text("Complete!"); bar.empty(); status_area.empty()
        result = self._combine(analyses)
        result["pe_parsed"]       = pe.result
        result["forensics"]       = forensics
        result["injection"]       = injection
        result["pe_text"]         = pe_text
        result["forensics_text"]  = forensics_text
        result["injection_text"]  = injection_text
        return result

    def _format_pe(self, pe: PEParser) -> str:
        if not pe.valid:
            return f"NOT A VALID PE FILE. Error: {pe.result.get('error','Unknown')}"
        r = pe.result
        lines = [
            f"Architecture:    {r.get('architecture')}",
            f"Type:            {r.get('pe_type')}",
            f"Entry Point:     {r.get('entry_point')}",
            f"Image Base:      {r.get('image_base')}",
            f"Size of Image:   {r.get('size_of_image')}",
            f"Subsystem:       {r.get('subsystem')}",
            f"Timestamp:       {r.get('timestamp_human')}",
            f"Characteristics: {', '.join(r.get('characteristics',[]))}",
            f"DLL Chars:       {', '.join(r.get('dll_characteristics',[]))}",
            "",
            "SECTIONS:",
        ]
        for s in r.get("sections", []):
            flag = "⚠️ SUSPICIOUS" if s.get("suspicious") else ""
            lines.append(f"  {s['name']:10} VA=0x{s['virtual_address']:08X} "
                         f"entropy={s['entropy']} flags={s['flags']} {flag}")
        lines.append("\nIMPORTS (top 60 APIs per DLL):")
        for dll, funcs in r.get("imports", {}).items():
            lines.append(f"  [{dll}]: {', '.join(funcs[:60])}")
        exports = r.get("exports", [])
        if exports:
            lines.append(f"\nEXPORTS: {', '.join(exports[:20])}")
        return "\n".join(lines)

    def _format_forensics(self, f: Dict) -> str:
        lines = [
            f"Entropy:          {f['overall_entropy']} — {f['entropy_verdict']}",
            f"Shellcode sigs:   {f.get('shellcode_signatures') or 'None'}",
            f"EXITFUNC markers: {f.get('exitfunc_markers') or 'None'}",
            f"NOP sled:         {f.get('nop_sled') or 'None'}",
            f"Encoder artifacts:{f.get('encoder_artifacts') or 'None'}",
            f"Suspicious sects: {f.get('suspicious_sections') or 'None'}",
            f"Embedded MZ hdrs: {f.get('embedded_pe_headers') or 'None'}",
        ]
        return "\n".join(lines)

    def _format_injection(self, inj: Dict) -> str:
        lines = [f"Risk Score: {inj['risk_score']}"]
        if inj["detected_techniques"]:
            lines.append("DETECTED INJECTION TECHNIQUES:")
            for t in inj["detected_techniques"]:
                lines.append(f"  [{t['severity']}][confidence:{t['confidence']}] {t['technique']}")
                lines.append(f"    APIs: {', '.join(t['matched_apis'])}")
                lines.append(f"    Desc: {t['description']}")
        else:
            lines.append("No injection techniques detected.")
        if inj["suspicious_api_combinations"]:
            lines.append(f"SUSPICIOUS API COMBOS: {inj['suspicious_api_combinations']}")
        if inj["evasion_capabilities"]:
            lines.append("EVASION CAPABILITIES:")
            for cat, apis in inj["evasion_capabilities"].items():
                lines.append(f"  {cat}: {', '.join(apis)}")
        return "\n".join(lines)

    # ── report ─────────────────────────────────────────────────────────────────
    def create_report(self, results: Dict, title: str = "Security Analysis Report") -> str:
        doc = Document()
        doc.add_heading(title, 0)
        tl = results.get("threat_level","UNKNOWN")
        doc.add_heading(f"Threat Level: {tl}", level=1)
        if results.get("verdict"):
            doc.add_paragraph(results["verdict"], style='Body Text')

        if results.get("injection"):
            doc.add_heading("Code Injection Analysis", level=1)
            doc.add_paragraph(results.get("injection_text",""), style='Body Text')

        if results.get("pe_text"):
            doc.add_heading("PE Structure Analysis", level=1)
            doc.add_paragraph(results.get("pe_text",""), style='Body Text')

        if results.get("forensics_text"):
            doc.add_heading("Binary Forensics", level=1)
            doc.add_paragraph(results.get("forensics_text",""), style='Body Text')

        doc.add_heading("Executive Summary", level=1)
        for pt in results.get("summary",["No summary"]):
            doc.add_paragraph(pt, style='Body Text')

        for sec, content in results.get("sections",{}).items():
            doc.add_heading(sec.replace('_',' ').title(), level=1)
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
            doc.add_heading("Errors", level=1)
            for e in results["errors"]:
                doc.add_paragraph(f"• {e}", style='List Bullet')

        fname = f"security_report_{os.getpid()}.docx"
        doc.save(fname)
        return fname

    def get_chat_response(self, user_input: str) -> str:
        return self.conversation.predict(input=user_input + " Be short and crisp.")


# ══════════════════════════════════════════════════════════════════════════════
#  DISPLAY
# ══════════════════════════════════════════════════════════════════════════════

THREAT_COLOR = {"HIGH":"#ff4444","MEDIUM":"#ffaa00","LOW":"#44ff88","CLEAN":"#44ff88","UNKNOWN":"#9D4EDD"}

def display_threat_banner(results: Dict):
    tl = results.get("threat_level","UNKNOWN")
    color = THREAT_COLOR.get(tl, "#9D4EDD")
    st.markdown(
        f"<div style='background:#1a0a2e;border-left:6px solid {color};"
        f"border-radius:8px;padding:16px;margin:12px 0;'>"
        f"<h3 style='color:{color};margin:0'>⚠️ Threat Level: {tl}</h3>"
        f"<p style='color:#ccc;margin:4px 0 0'>{results.get('verdict','')}</p></div>",
        unsafe_allow_html=True,
    )

def display_pe_section(results: Dict):
    pe = results.get("pe_parsed")
    if not pe or not pe.get("architecture"):
        return
    with st.expander("🗂️ PE Structure", expanded=True):
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Arch",       pe.get("architecture","?"))
        c2.metric("Type",       pe.get("pe_type","?"))
        c3.metric("Entry Point",pe.get("entry_point","?"))
        c4.metric("Subsystem",  pe.get("subsystem","?"))
        st.caption(f"Timestamp: {pe.get('timestamp_human','?')} | "
                   f"Chars: {', '.join(pe.get('characteristics',[]))}")
        st.caption(f"DLL Chars: {', '.join(pe.get('dll_characteristics',[]))}")

        st.markdown("**Sections:**")
        for s in pe.get("sections",[]):
            badge = "🔴" if s.get("suspicious") else "🟢"
            st.markdown(
                f"{badge} `{s['name']:10}` "
                f"VA=`{s['virtual_address']:08X}` "
                f"entropy=`{s['entropy']}` "
                f"flags=`{', '.join(s['flags'])}`"
            )

        if pe.get("imports"):
            st.markdown("**Imports:**")
            for dll, funcs in pe["imports"].items():
                with st.expander(f"  {dll} ({len(funcs)} functions)"):
                    st.code(", ".join(funcs))

        if pe.get("exports"):
            st.markdown(f"**Exports:** {', '.join(pe['exports'][:20])}")

def display_injection_section(results: Dict):
    inj = results.get("injection")
    if not inj:
        return
    with st.expander("🎯 Code Injection Detection", expanded=True):
        score = inj.get("risk_score", 0)
        color = "#ff4444" if score >= 5 else "#ffaa00" if score >= 2 else "#44ff88"
        st.markdown(
            f"<div style='background:#1a0a2e;border-left:4px solid {color};"
            f"padding:8px;border-radius:6px;'>"
            f"<b style='color:{color}'>Risk Score: {score}</b></div>",
            unsafe_allow_html=True,
        )
        techs = inj.get("detected_techniques",[])
        if techs:
            st.error(f"🚨 {len(techs)} injection technique(s) detected!")
            for t in techs:
                sev_color = "#ff4444" if t["severity"]=="HIGH" else "#ffaa00"
                st.markdown(
                    f"<div style='background:#1a0a2e;border-left:4px solid {sev_color};"
                    f"padding:10px;border-radius:6px;margin:6px 0;'>"
                    f"<b style='color:{sev_color}'>{t['technique']}</b> "
                    f"[{t['severity']}] confidence: {t['confidence']}<br>"
                    f"<small>{t['description']}</small><br>"
                    f"<code>{', '.join(t['matched_apis'])}</code></div>",
                    unsafe_allow_html=True,
                )
        else:
            st.success("No injection techniques detected in import table.")

        combos = inj.get("suspicious_api_combinations",[])
        if combos:
            st.warning("⚠️ Suspicious API combinations:")
            for c in combos:
                st.markdown(f"- `{c}`")

        evasion = inj.get("evasion_capabilities",{})
        if evasion:
            st.markdown("**Evasion capabilities:**")
            for cat, apis in evasion.items():
                st.markdown(f"- **{cat.replace('_',' ').title()}**: `{', '.join(apis)}`")

def display_forensics_section(results: Dict):
    f = results.get("forensics")
    if not f:
        return
    with st.expander("🔬 Binary Forensics", expanded=True):
        c1,c2 = st.columns(2)
        c1.metric("Entropy", f.get("overall_entropy","?"))
        c2.write(f.get("entropy_verdict",""))

        if f.get("shellcode_signatures"):
            st.error("🚨 Shellcode Signatures Found!")
            for s in f["shellcode_signatures"]:
                st.code(s)
        if f.get("exitfunc_markers"):
            st.error(f"🚨 EXITFUNC (msfvenom indicator): {f['exitfunc_markers']}")
        if f.get("nop_sled"):
            st.warning(f"⚠️ NOP sled at {f['nop_sled']}")
        if f.get("encoder_artifacts"):
            st.warning(f"⚠️ Encoder artifacts: {', '.join(f['encoder_artifacts'])}")
        if f.get("suspicious_sections"):
            st.warning("⚠️ Suspicious sections:")
            for s in f["suspicious_sections"]:
                st.markdown(f"- `{s}`")
        if f.get("embedded_pe_headers"):
            st.warning(f"⚠️ Embedded PE/MZ headers at: {f['embedded_pe_headers']}")

def display_analysis_results(results: Dict):
    display_threat_banner(results)
    display_pe_section(results)
    display_injection_section(results)
    display_forensics_section(results)

    st.header("Executive Summary")
    for pt in results.get("summary",["No summary"]):
        st.write(pt)
    st.divider()

    if results.get("errors"):
        st.error("Errors")
        for e in results["errors"]: st.write(f"• {e}")
        st.divider()

    for sec_name, content in results.get("sections",{}).items():
        st.subheader(sec_name.replace('_',' ').title())
        if content.get("description"): st.write(content["description"])
        for finding in content.get("findings",[]): st.write(f"• {finding}")
        st.divider()

def download_report_button(fname: str, label: str = "📥 Download Report"):
    with open(fname,"rb") as f:
        st.download_button(label=label, data=f, file_name=fname,
                           mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    os.remove(fname)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    groq_api_key = api_key_sidebar()
    with st.sidebar:
        st.write("---")
        tabs = on_hover_tabs(
            tabName=['Code Analyzer','GAMKERSGPT','About'],
            iconName=['code','chat','info'],
            styles={
                'navtab':{'background-color':'black','color':'#9D4EDD','font-size':'16px',
                          'transition':'.3s','white-space':'nowrap','text-transform':'uppercase'},
                'tabOptionsStyle':{':hover':{'color':'#1A1A1A','background-color':'black'}},
                'iconStyle':{'position':'fixed','left':'7.5px','text-align':'left'},
                'tabStyle':{'list-style-type':'none','margin-bottom':'30px','padding-left':'30px'},
            },
        )

    if tabs == 'About':
        about_section(); return

    if not groq_api_key:
        st.info("👈 Enter your Groq API key in the sidebar to get started."); return

    if "app" not in st.session_state or st.session_state.get("_api_key") != groq_api_key:
        st.session_state.app      = SecurityAnalysisApp(groq_api_key)
        st.session_state.messages = []
        st.session_state._api_key = groq_api_key

    # ── Code Analyzer ──────────────────────────────────────────────────────────
    if tabs == 'Code Analyzer':
        st.title("GAMKERS Security Analyzer")
        tab1, tab2, tab3 = st.tabs(["📝 Paste Code","📁 Upload Source","💾 Upload Binary"])

        with tab1:
            code_input = st.text_area("Paste your code here:", height=300)
            if st.button("🔍 Analyze Code", key="analyze_pasted") and code_input:
                with st.spinner("🔄 Analyzing..."):
                    results = st.session_state.app.analyze_code(code_input)
                    display_analysis_results(results)
                    download_report_button(st.session_state.app.create_report(results))

        with tab2:
            uploaded_file = st.file_uploader("Choose source code file",
                type=['py','js','java','cpp','cs','php','rb'], key="code_file")
            if st.button("🔍 Analyze Source", key="analyze_source") and uploaded_file:
                with st.spinner("🔄 Analyzing..."):
                    results = st.session_state.app.analyze_code(uploaded_file.read().decode())
                    display_analysis_results(results)
                    download_report_button(st.session_state.app.create_report(results))

        with tab3:
            st.write("Upload a PE binary (.exe/.dll/.bin) for full static analysis")
            uploaded_binary = st.file_uploader("Choose binary file",
                type=['exe','dll','bin'], key="binary_file")
            if uploaded_binary:
                st.info("Runs: PE parsing → entropy analysis → injection detection → AI verdict")
            if st.button("🔍 Analyze Binary", key="analyze_binary") and uploaded_binary:
                binary_data = uploaded_binary.read()
                with st.expander("📄 Extracted Strings Preview"):
                    preview = st.session_state.app.extract_strings(binary_data)
                    st.text_area("Strings", value="\n".join(preview)[:10000] +
                                 ("\n[Truncated...]" if len(preview)>200 else ""),
                                 height=200, disabled=True)
                results = st.session_state.app.analyze_binary(binary_data)
                st.subheader("Binary Analysis Results")
                display_analysis_results(results)
                download_report_button(
                    st.session_state.app.create_report(results,
                        title=f"Binary Analysis — {uploaded_binary.name}"),
                    label="📥 Download Binary Report",
                )

    # ── GAMKERSGPT ─────────────────────────────────────────────────────────────
    elif tabs == 'GAMKERSGPT':
        st.title("GAMKERSGPT - Security Assistant")
        st.markdown("""<style>
            [data-testid="stChatMessage"]{background-color:#000!important;color:#fff!important;padding:10px;border-radius:8px;}
            [data-testid="stChatInput"]{background-color:#000!important;color:#fff!important;border:1px solid #333;border-radius:8px;}
            [data-testid="stBottom"]{background-color:#000!important;padding:10px!important;border-radius:8px!important;}
            </style>""", unsafe_allow_html=True)
        st.markdown("### Welcome to GAMKERSGPT!")
        st.write("Ask about vulnerabilities, injection techniques, malware analysis, and more.")
        st.info("Tip: Be as specific as possible.")

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("Ask your cyber security question..."):
            st.session_state.messages.append({"role":"user","content":prompt})
            with st.chat_message("user"): st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("🤔 Thinking..."):
                    response = st.session_state.app.get_chat_response(prompt)
                    response = response.split("</think>")[-1].strip()
                    st.markdown(response)
                    st.session_state.messages.append({"role":"assistant","content":response})

if __name__ == "__main__":
    main()
