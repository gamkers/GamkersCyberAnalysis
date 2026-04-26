import streamlit as st
import json, math, struct, re, os, subprocess, tempfile, hashlib, time
from typing import List, Dict, Tuple, Optional
from collections import Counter
from langchain_groq import ChatGroq
from langchain_classic.chains import LLMChain, ConversationChain
from langchain_classic.prompts import PromptTemplate
from langchain_classic.memory import ConversationBufferMemory
from docx import Document

# ──────────────────────────────────────────────────────────────────────────────
#  STREAMLIT PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SentinelX SOC",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
#  GLOBAL CSS — SOC TERMINAL AESTHETIC
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&family=Orbitron:wght@700;900&display=swap');

:root {
  --bg:        #050a0f;
  --panel:     #080f17;
  --border:    #0d2137;
  --accent:    #00d4ff;
  --accent2:   #ff3c6e;
  --accent3:   #00ff88;
  --warn:      #ffaa00;
  --dim:       #1a3a52;
  --text:      #c8e6f5;
  --text-dim:  #4a7a99;
  --mono:      'Share Tech Mono', monospace;
  --display:   'Orbitron', sans-serif;
  --body:      'Rajdhani', sans-serif;
}

* { box-sizing: border-box; }

html, body, .stApp {
  background: var(--bg) !important;
  color: var(--text) !important;
  font-family: var(--body) !important;
}

/* scanline overlay */
.stApp::before {
  content: '';
  position: fixed;
  inset: 0;
  background: repeating-linear-gradient(
    0deg, transparent, transparent 2px,
    rgba(0,212,255,0.015) 2px, rgba(0,212,255,0.015) 4px
  );
  pointer-events: none;
  z-index: 9999;
}

/* sidebar */
[data-testid="stSidebar"] {
  background: var(--panel) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* all headings */
h1, h2, h3, h4 { font-family: var(--display) !important; color: var(--accent) !important; letter-spacing: 2px; }

/* tabs */
[data-testid="stTabs"] button {
  font-family: var(--mono) !important;
  color: var(--text-dim) !important;
  background: transparent !important;
  border-bottom: 2px solid transparent !important;
  font-size: 12px !important;
  letter-spacing: 1px;
}
[data-testid="stTabs"] button[aria-selected="true"] {
  color: var(--accent) !important;
  border-bottom-color: var(--accent) !important;
}

/* inputs */
.stTextArea textarea, .stTextInput input {
  background: var(--panel) !important;
  border: 1px solid var(--dim) !important;
  color: var(--text) !important;
  font-family: var(--mono) !important;
  font-size: 13px !important;
  border-radius: 2px !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 8px rgba(0,212,255,0.2) !important;
}

/* buttons */
.stButton > button {
  background: transparent !important;
  border: 1px solid var(--accent) !important;
  color: var(--accent) !important;
  font-family: var(--mono) !important;
  font-size: 12px !important;
  letter-spacing: 2px !important;
  border-radius: 2px !important;
  transition: all .2s !important;
  text-transform: uppercase !important;
}
.stButton > button:hover {
  background: var(--accent) !important;
  color: var(--bg) !important;
  box-shadow: 0 0 16px rgba(0,212,255,0.4) !important;
}

/* expander */
[data-testid="stExpander"] {
  background: var(--panel) !important;
  border: 1px solid var(--border) !important;
  border-radius: 2px !important;
}
[data-testid="stExpander"] summary { font-family: var(--mono) !important; color: var(--accent) !important; }

/* metrics */
[data-testid="stMetric"] {
  background: var(--panel) !important;
  border: 1px solid var(--border) !important;
  border-top: 2px solid var(--accent) !important;
  padding: 12px !important;
  border-radius: 2px !important;
}
[data-testid="stMetricLabel"] { font-family: var(--mono) !important; font-size: 11px !important; color: var(--text-dim) !important; }
[data-testid="stMetricValue"] { font-family: var(--display) !important; color: var(--accent) !important; font-size: 22px !important; }

/* progress */
[data-testid="stProgress"] > div > div > div > div { background: var(--accent) !important; }

/* file uploader */
[data-testid="stFileUploader"] {
  background: var(--panel) !important;
  border: 1px dashed var(--dim) !important;
  border-radius: 2px !important;
}

/* alerts */
.stAlert { border-radius: 2px !important; font-family: var(--mono) !important; font-size: 12px !important; }

/* chat messages */
[data-testid="stChatMessage"] {
  background: var(--panel) !important;
  border: 1px solid var(--border) !important;
  border-radius: 2px !important;
  font-family: var(--mono) !important;
  font-size: 13px !important;
}
[data-testid="stChatInput"] textarea {
  background: var(--panel) !important;
  border: 1px solid var(--dim) !important;
  color: var(--text) !important;
  font-family: var(--mono) !important;
}

/* code blocks */
code, pre {
  background: #020810 !important;
  color: var(--accent3) !important;
  font-family: var(--mono) !important;
  border: 1px solid var(--border) !important;
  border-radius: 2px !important;
}

/* hide streamlit chrome */
#MainMenu, footer, header { visibility: hidden !important; }
[data-testid="stToolbar"] { display: none !important; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
#  UI COMPONENTS
# ──────────────────────────────────────────────────────────────────────────────

def soc_header():
    st.markdown("""
    <div style="display:flex;align-items:center;gap:16px;padding:12px 0 20px;border-bottom:1px solid #0d2137;margin-bottom:20px;">
      <div style="font-family:'Orbitron',sans-serif;font-size:26px;font-weight:900;
                  color:#00d4ff;letter-spacing:4px;text-shadow:0 0 20px rgba(0,212,255,0.5);">
        ⬡ SentinelX
      </div>
      <div style="font-family:'Share Tech Mono',monospace;font-size:11px;color:#4a7a99;
                  border-left:1px solid #0d2137;padding-left:16px;line-height:1.8;">
        MALWARE ANALYSIS PLATFORM v2.0<br>
        <span style="color:#00ff88;">● SYSTEMS NOMINAL</span>
        &nbsp;&nbsp;
        <span style="color:#ffaa00;">◈ AI ENGINE READY</span>
      </div>
      <div style="margin-left:auto;font-family:'Share Tech Mono',monospace;font-size:10px;color:#1a3a52;text-align:right;">
        GAMKERS SECURITY OPS<br>
        CLASSIFIED — ANALYST ONLY
      </div>
    </div>
    """, unsafe_allow_html=True)

def threat_badge(level: str, score: int = 0):
    colors = {"HIGH": "#ff3c6e", "MEDIUM": "#ffaa00", "LOW": "#00ff88",
              "CLEAN": "#00ff88", "CRITICAL": "#ff3c6e", "UNKNOWN": "#4a7a99"}
    color = colors.get(level, "#4a7a99")
    glyphs = {"HIGH": "▲▲▲", "CRITICAL": "▲▲▲▲", "MEDIUM": "▲▲◻", "LOW": "▲◻◻", "CLEAN": "◻◻◻", "UNKNOWN": "?"}
    glyph  = glyphs.get(level, "?")
    st.markdown(f"""
    <div style="background:#080f17;border:1px solid {color};border-left:4px solid {color};
                padding:16px 20px;border-radius:2px;margin:12px 0;
                box-shadow:0 0 20px {color}22;">
      <div style="display:flex;align-items:center;gap:16px;">
        <div style="font-family:'Orbitron',sans-serif;font-size:28px;font-weight:900;
                    color:{color};text-shadow:0 0 12px {color}88;">{glyph}</div>
        <div>
          <div style="font-family:'Orbitron',sans-serif;font-size:13px;
                      color:{color};letter-spacing:3px;">THREAT LEVEL</div>
          <div style="font-family:'Orbitron',sans-serif;font-size:22px;
                      font-weight:900;color:{color};">{level}</div>
        </div>
        {"<div style='margin-left:auto;font-family:\"Share Tech Mono\",monospace;font-size:28px;color:"+color+";'>RISK: "+str(score)+"</div>" if score else ""}
      </div>
    </div>
    """, unsafe_allow_html=True)

def stat_row(items: list):
    """items = [(label, value, color), ...]"""
    cols = st.columns(len(items))
    for col, (label, value, color) in zip(cols, items):
        col.markdown(f"""
        <div style="background:#080f17;border:1px solid #0d2137;border-top:2px solid {color};
                    padding:14px;text-align:center;">
          <div style="font-family:'Share Tech Mono',monospace;font-size:10px;color:#4a7a99;
                      letter-spacing:1px;margin-bottom:6px;">{label}</div>
          <div style="font-family:'Orbitron',sans-serif;font-size:18px;font-weight:700;color:{color};">{value}</div>
        </div>
        """, unsafe_allow_html=True)

def panel_header(title: str, subtitle: str = "", color: str = "#00d4ff"):
    st.markdown(f"""
    <div style="border-left:3px solid {color};padding:4px 12px;margin:16px 0 10px;
                background:linear-gradient(90deg,{color}11,transparent);">
      <div style="font-family:'Orbitron',sans-serif;font-size:12px;color:{color};
                  letter-spacing:2px;">{title}</div>
      {"<div style='font-family:Share Tech Mono,monospace;font-size:10px;color:#4a7a99;'>"+subtitle+"</div>" if subtitle else ""}
    </div>
    """, unsafe_allow_html=True)

def finding_card(text: str, severity: str = "info"):
    colors = {"critical": "#ff3c6e", "high": "#ff6b35", "medium": "#ffaa00",
              "low": "#00ff88", "info": "#00d4ff"}
    color = colors.get(severity.lower(), "#00d4ff")
    st.markdown(f"""
    <div style="background:#050a0f;border-left:2px solid {color};padding:8px 12px;
                margin:3px 0;font-family:'Share Tech Mono',monospace;font-size:11px;color:#c8e6f5;">
      <span style="color:{color};">▸</span> {text}
    </div>
    """, unsafe_allow_html=True)

def technique_card(name: str, severity: str, confidence: str, desc: str, apis: list):
    sev_color = {"HIGH": "#ff3c6e", "MEDIUM": "#ffaa00", "LOW": "#00ff88"}.get(severity, "#4a7a99")
    conf_color = {"HIGH": "#00ff88", "MEDIUM": "#ffaa00", "LOW": "#ff6b35"}.get(confidence, "#4a7a99")
    st.markdown(f"""
    <div style="background:#050a0f;border:1px solid #0d2137;border-left:3px solid {sev_color};
                padding:12px;margin:6px 0;border-radius:2px;">
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">
        <span style="font-family:'Orbitron',sans-serif;font-size:11px;color:{sev_color};
                     font-weight:700;">{name}</span>
        <span style="font-family:'Share Tech Mono',monospace;font-size:9px;
                     background:{sev_color}22;color:{sev_color};padding:2px 6px;">{severity}</span>
        <span style="font-family:'Share Tech Mono',monospace;font-size:9px;
                     background:{conf_color}22;color:{conf_color};padding:2px 6px;margin-left:auto;">
          CONF: {confidence}
        </span>
      </div>
      <div style="font-family:'Share Tech Mono',monospace;font-size:10px;color:#4a7a99;margin-bottom:6px;">{desc}</div>
      <div style="font-family:'Share Tech Mono',monospace;font-size:10px;color:#00d4ff;word-break:break-all;">
        APIs: {', '.join(apis[:8])}{"..." if len(apis)>8 else ""}
      </div>
    </div>
    """, unsafe_allow_html=True)

def entropy_bar(value: float):
    pct = min(100, (value / 8.0) * 100)
    color = "#ff3c6e" if value > 7 else "#ffaa00" if value > 6 else "#00ff88"
    st.markdown(f"""
    <div style="margin:8px 0;">
      <div style="display:flex;justify-content:space-between;font-family:'Share Tech Mono',monospace;
                  font-size:10px;color:#4a7a99;margin-bottom:4px;">
        <span>ENTROPY</span><span style="color:{color};">{value:.3f} / 8.0</span>
      </div>
      <div style="height:6px;background:#0d2137;border-radius:1px;">
        <div style="height:100%;width:{pct:.1f}%;background:{color};
                    box-shadow:0 0 8px {color}88;border-radius:1px;transition:width .5s;"></div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PE PARSER (enhanced)
# ══════════════════════════════════════════════════════════════════════════════
class PEParser:
    MACHINE_TYPES = {
        0x0000:"Unknown", 0x014C:"x86 (32-bit)", 0x0200:"IA-64",
        0x8664:"x86-64 (64-bit)", 0x01C4:"ARMv7",  0xAA64:"ARM64",
    }
    SECTION_FLAGS = {
        0x00000020:"CODE", 0x00000040:"INITIALIZED_DATA",
        0x00000080:"UNINITIALIZED_DATA", 0x20000000:"EXECUTE",
        0x40000000:"READ", 0x80000000:"WRITE",
    }
    CHARACTERISTICS = {
        0x0001:"NO_RELOCS", 0x0002:"EXECUTABLE", 0x0004:"NO_LINE_NUMS",
        0x0020:"LARGE_ADDR_AWARE", 0x0100:"32BIT_MACHINE",
        0x1000:"SYSTEM", 0x2000:"DLL", 0x4000:"UP_SYSTEM_ONLY",
    }
    DLL_CHARACTERISTICS = {
        0x0020:"HIGH_ENTROPY_VA", 0x0040:"DYNAMIC_BASE (ASLR)",
        0x0080:"FORCE_INTEGRITY", 0x0100:"NX_COMPAT (DEP)",
        0x0400:"NO_ISOLATION", 0x0800:"NO_SEH",
        0x1000:"NO_BIND", 0x4000:"WDM_DRIVER", 0x8000:"TERMINAL_SERVER",
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

    def _u8(self,o):  return struct.unpack_from("B",self.data,o)[0]
    def _u16(self,o): return struct.unpack_from("<H",self.data,o)[0]
    def _u32(self,o): return struct.unpack_from("<I",self.data,o)[0]
    def _u64(self,o): return struct.unpack_from("<Q",self.data,o)[0]

    def _rva_to_offset(self, rva: int) -> Optional[int]:
        for s in self.sections:
            va = s["virtual_address"]; size = s["virtual_size"] or s["raw_size"]
            if va <= rva < va + size:
                return s["raw_offset"] + (rva - va)
        return None

    def _read_string(self, offset: int, max_len: int = 256) -> str:
        end = self.data.find(b'\x00', offset, offset + max_len)
        if end == -1: end = offset + max_len
        return self.data[offset:end].decode('latin-1', errors='replace')

    def _parse(self):
        d = self.data; r = {}
        try:
            if d[:2] != b'MZ':
                self.result = {"error": "Not a PE file (missing MZ)"}; return
            self.pe_offset = self._u32(0x3C); pe = self.pe_offset
            if d[pe:pe+4] != b'PE\x00\x00':
                self.result = {"error": "Missing PE signature"}; return
            self.valid = True
            machine   = self._u16(pe+4); num_sects = self._u16(pe+6)
            timestamp = self._u32(pe+8); opt_size  = self._u16(pe+20)
            chars     = self._u16(pe+22)
            r["architecture"]    = self.MACHINE_TYPES.get(machine, f"0x{machine:04X}")
            r["num_sections"]    = num_sects
            r["timestamp"]       = timestamp
            r["timestamp_human"] = self._ts(timestamp)
            r["characteristics"] = [v for k,v in self.CHARACTERISTICS.items() if chars & k]
            opt = pe + 24; magic = self._u16(opt)
            self.is_64bit = (magic == 0x20B)
            r["is_64bit"] = self.is_64bit
            r["pe_type"]  = "PE32+" if self.is_64bit else "PE32"
            r["entry_point"]     = f"0x{self._u32(opt+16):08X}"
            r["image_base"]      = f"0x{(self._u64(opt+24) if self.is_64bit else self._u32(opt+28)):016X}"
            r["size_of_image"]   = f"0x{self._u32(opt+56):08X}"
            r["size_of_headers"] = f"0x{self._u32(opt+60):08X}"
            r["subsystem"]       = {1:"Native",2:"Windows GUI",3:"Windows CUI",
                                     9:"WinCE GUI",14:"EFI",16:"EFI Runtime"}.get(self._u16(opt+68),"Unknown")
            dll_chars = self._u16(opt+70)
            r["dll_characteristics"] = [v for k,v in self.DLL_CHARACTERISTICS.items() if dll_chars & k]
            dd_offset = opt + (112 if self.is_64bit else 96)
            sect_offset = opt + opt_size; sections = []
            for i in range(num_sects):
                base = sect_offset + i * 40
                name = d[base:base+8].rstrip(b'\x00').decode('latin-1', errors='replace')
                vsz  = self._u32(base+8); va   = self._u32(base+12)
                rsz  = self._u32(base+16); roff = self._u32(base+20)
                sc   = self._u32(base+36)
                flags = [v for k,v in self.SECTION_FLAGS.items() if sc & k]
                raw   = d[roff:roff+rsz] if roff and rsz else b''
                ent   = self._entropy(raw)
                sections.append({
                    "name":name,"virtual_address":va,"virtual_size":vsz,
                    "raw_offset":roff,"raw_size":rsz,"flags":flags,
                    "entropy":round(ent,3),
                    "suspicious": ent > 7.0 or ("EXECUTE" in flags and "WRITE" in flags),
                })
            self.sections = sections; r["sections"] = sections
            imp_rva = self._u32(dd_offset)
            if imp_rva:
                r["imports"] = self._parse_imports(imp_rva); self.imports = r["imports"]
            exp_rva_real = self._u32(dd_offset - 8) if dd_offset >= 8 else 0
            r["exports"] = self._parse_exports(exp_rva_real) if exp_rva_real else []
            self.exports = r["exports"]
        except Exception as ex:
            r["parse_error"] = str(ex)
        self.result = r

    def _parse_imports(self, imp_rva):
        imports = {}
        try:
            off = self._rva_to_offset(imp_rva)
            if off is None: return imports
            while True:
                ilt_rva = self._u32(off); name_rva = self._u32(off+12); iat_rva = self._u32(off+16)
                if name_rva == 0: break
                name_off = self._rva_to_offset(name_rva)
                dll_name = self._read_string(name_off) if name_off else "Unknown"
                funcs = []; lookup_rva = ilt_rva or iat_rva
                if lookup_rva:
                    loff = self._rva_to_offset(lookup_rva)
                    if loff:
                        while True:
                            entry = self._u64(loff) if self.is_64bit else self._u32(loff)
                            loff += 8 if self.is_64bit else 4
                            if entry == 0: break
                            ordinal_flag = (1<<63) if self.is_64bit else (1<<31)
                            if entry & ordinal_flag:
                                funcs.append(f"Ordinal#{entry & 0xFFFF}")
                            else:
                                fn_off = self._rva_to_offset(entry & 0x7FFFFFFF)
                                if fn_off: funcs.append(self._read_string(fn_off+2))
                imports[dll_name] = funcs; off += 20
        except: pass
        return imports

    def _parse_exports(self, exp_rva):
        exports = []
        try:
            off = self._rva_to_offset(exp_rva)
            if off is None: return exports
            num_names = self._u32(off+24); names_rva = self._u32(off+32)
            names_off = self._rva_to_offset(names_rva)
            if names_off is None: return exports
            for i in range(min(num_names, 500)):
                n_rva = self._u32(names_off+i*4)
                n_off = self._rva_to_offset(n_rva)
                if n_off: exports.append(self._read_string(n_off))
        except: pass
        return exports

    @staticmethod
    def _entropy(data: bytes) -> float:
        if not data: return 0.0
        c = Counter(data); l = len(data)
        return -sum((v/l)*math.log2(v/l) for v in c.values())

    @staticmethod
    def _ts(ts: int) -> str:
        import datetime
        try: return datetime.datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S UTC')
        except: return "Invalid"


# ══════════════════════════════════════════════════════════════════════════════
#  INJECTION DETECTOR (enhanced)
# ══════════════════════════════════════════════════════════════════════════════
class InjectionDetector:
    TECHNIQUES = [
        {"name":"Process Hollowing","apis":["NtUnmapViewOfSection","ZwUnmapViewOfSection","SetThreadContext","GetThreadContext","ResumeThread"],"also_needs":["VirtualAllocEx","WriteProcessMemory"],"description":"Hollows legitimate process, injects malicious PE","severity":"HIGH"},
        {"name":"Classic DLL Injection","apis":["LoadLibraryA","LoadLibraryW"],"also_needs":["CreateRemoteThread","VirtualAllocEx","WriteProcessMemory"],"description":"Forces target to load attacker DLL via CreateRemoteThread","severity":"HIGH"},
        {"name":"Reflective DLL Injection","apis":["ReflectiveLoader","ReflectiveDllInjection"],"also_needs":[],"description":"DLL loads itself from memory, evades LoadLibrary detection","severity":"HIGH"},
        {"name":"Shellcode / PE Injection","apis":["VirtualAllocEx","WriteProcessMemory"],"also_needs":["CreateRemoteThread","NtCreateThreadEx","RtlCreateUserThread"],"description":"Allocates remote memory, writes and executes shellcode","severity":"HIGH"},
        {"name":"APC Injection","apis":["QueueUserAPC","NtQueueApcThread","NtQueueApcThreadEx"],"also_needs":[],"description":"Queues APC to hijack thread execution in target process","severity":"HIGH"},
        {"name":"Atom Bombing","apis":["GlobalAddAtomA","GlobalAddAtomW","GlobalGetAtomNameA"],"also_needs":["NtQueueApcThread"],"description":"Uses Windows atom tables to smuggle shellcode","severity":"HIGH"},
        {"name":"Section / Shared Memory Injection","apis":["NtCreateSection","ZwCreateSection","NtMapViewOfSection","ZwMapViewOfSection"],"also_needs":[],"description":"Creates shared memory section mapped into target","severity":"HIGH"},
        {"name":"SetWindowsHookEx Injection","apis":["SetWindowsHookExA","SetWindowsHookExW"],"also_needs":[],"description":"Message hook forces DLL into target processes","severity":"MEDIUM"},
        {"name":"IAT Hooking","apis":["VirtualProtect"],"also_needs":["GetProcAddress","GetModuleHandleA","GetModuleHandleW"],"description":"Modifies IAT to redirect API calls","severity":"MEDIUM"},
        {"name":"Early Bird APC","apis":["CreateProcessA","CreateProcessW"],"also_needs":["VirtualAllocEx","WriteProcessMemory","QueueUserAPC"],"description":"Injects APC before process main thread initializes","severity":"HIGH"},
        {"name":"Process Doppelgänging","apis":["NtCreateTransaction","NtWriteFile","NtCreateSection","NtRollbackTransaction"],"also_needs":[],"description":"NTFS transactions to create process from phantom file","severity":"HIGH"},
        {"name":"Thread Execution Hijacking","apis":["SuspendThread","SetThreadContext","ResumeThread"],"also_needs":["VirtualAllocEx","WriteProcessMemory"],"description":"Suspends thread, redirects EIP/RIP to shellcode","severity":"HIGH"},
        {"name":"Heaven's Gate (WoW64)","apis":["Wow64Transition","wow64cpu","NtWow64QueryInformationProcess64"],"also_needs":[],"description":"32-bit process switches to 64-bit mode to bypass hooks","severity":"HIGH"},
        {"name":"Module Stomping","apis":["LoadLibraryExA","LoadLibraryExW","MapViewOfFile"],"also_needs":["WriteProcessMemory","VirtualProtect"],"description":"Overwrites loaded module memory with shellcode","severity":"HIGH"},
        {"name":"Phantom DLL Hollowing","apis":["NtCreateSection","NtMapViewOfSection"],"also_needs":["NtUnmapViewOfSection","WriteProcessMemory"],"description":"Maps and hollows unloaded/phantom DLL into process","severity":"HIGH"},
        {"name":"Transacted Hollowing","apis":["NtCreateTransaction","NtCreateSection"],"also_needs":["NtMapViewOfSection","SetThreadContext"],"description":"Combines process hollowing with NTFS transactions","severity":"HIGH"},
    ]
    EVASION_APIS = {
        "anti_debug": ["IsDebuggerPresent","CheckRemoteDebuggerPresent","NtQueryInformationProcess","OutputDebugStringA","FindWindowA","FindWindowW","NtSetInformationThread","NtQuerySystemInformation"],
        "anti_vm":    ["GetSystemInfo","cpuid","vpcext","vmware","VirtualBox","VBOX","QEMU","GetComputerNameA","GetUserNameA"],
        "timing":     ["GetTickCount","GetTickCount64","QueryPerformanceCounter","timeGetTime","NtDelayExecution","Sleep","WaitForSingleObject"],
        "privilege":  ["AdjustTokenPrivileges","LookupPrivilegeValueA","SeDebugPrivilege","OpenProcessToken","ImpersonateLoggedOnUser","DuplicateTokenEx","RtlAdjustPrivilege"],
        "persistence":["RegSetValueExA","RegSetValueExW","RegCreateKeyExA","CreateServiceA","CreateServiceW","OpenSCManagerA","SchRpcRegisterTask","ITaskScheduler","SHGetSpecialFolderPath"],
        "network":    ["WSAStartup","connect","send","recv","HttpSendRequest","InternetOpenUrl","WinHttpConnect","URLDownloadToFile","socket","WSAConnect"],
        "credential": ["CredEnumerateA","CryptUnprotectData","LsaCallAuthenticationPackage","SamQueryInformationUser","NetUserEnum","NlpGetPrimaryCredential"],
        "sandbox":    ["GetCursorPos","GetForegroundWindow","BlockInput","GetLastInputInfo","GetAsyncKeyState","EnumWindows","FindWindowExA"],
    }

    @classmethod
    def detect(cls, imports: Dict[str,List[str]], string_list: List[str]) -> Dict:
        all_apis = set()
        for dll_funcs in imports.values():
            all_apis.update(f.lower() for f in dll_funcs)
        all_apis.update(s.lower() for s in string_list if len(s) < 60)
        results = {"detected_techniques":[],"evasion_capabilities":{},"suspicious_api_combinations":[],"risk_score":0}
        for technique in cls.TECHNIQUES:
            primary_hit   = [a for a in technique["apis"]       if a.lower() in all_apis]
            secondary_hit = [a for a in technique["also_needs"] if a.lower() in all_apis]
            if primary_hit:
                confidence = "HIGH" if (not technique["also_needs"] or secondary_hit) else "MEDIUM"
                results["detected_techniques"].append({
                    "technique": technique["name"], "severity": technique["severity"],
                    "confidence": confidence, "description": technique["description"],
                    "matched_apis": primary_hit + secondary_hit,
                })
                results["risk_score"] += 3 if confidence == "HIGH" else 1
        for category, apis in cls.EVASION_APIS.items():
            hits = [a for a in apis if a.lower() in all_apis]
            if hits:
                results["evasion_capabilities"][category] = hits
                results["risk_score"] += 1
        alloc  = "virtualallocex"     in all_apis
        write  = "writeprocessmemory" in all_apis
        thread = any(a in all_apis for a in ["createremotethread","ntcreatethreadex","rtlcreateuserthread"])
        if alloc and write and thread:
            results["suspicious_api_combinations"].append("Injection triad: VirtualAllocEx + WriteProcessMemory + CreateRemoteThread")
            results["risk_score"] += 5
        return results


# ══════════════════════════════════════════════════════════════════════════════
#  BINARY FORENSICS (enhanced)
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
        (b"\x55\x8b\xec\x83\xec",     "x86 function prologue (stdcall)"),
        (b"\x64\x8b\x35",             "TEB/PEB access via FS segment"),
        (b"\x65\x48\x8b\x04\x25",     "TEB/PEB access via GS segment (x64)"),
        (b"\xff\xd0",                 "CALL EAX — common shellcode exec"),
        (b"\xff\xd5",                 "CALL EBP — Metasploit style exec"),
    ]
    PACKER_SIGS = [
        (b"UPX0",       "UPX packer"),
        (b"UPX1",       "UPX packer"),
        (b"UPX2",       "UPX packer"),
        (b".aspack",    "ASPack packer"),
        (b"MPRESS",     "MPRESS packer"),
        (b"Petite",     "Petite packer"),
        (b"PECompact",  "PECompact packer"),
        (b"Nullsoft",   "NSIS installer"),
        (b"Themida",    "Themida protector"),
        (b"VMProtect",  "VMProtect obfuscator"),
        (b"Enigma",     "Enigma Protector"),
        (b"ExeCryptor", "ExeCryptor packer"),
    ]
    ENCODER_ARTIFACTS = [
        "shikata","EXITFUNC","metsrv","meterpreter","ReflectiveDll",
        "reverse_tcp","reverse_https","bind_tcp","migrate","getsystem",
        "hashdump","staged","stager","msf","cobalt_strike","beacon",
        "shellcode","loader","dropper","payload","inject","hook",
    ]
    C2_PATTERNS = [
        (r'\b(?:\d{1,3}\.){3}\d{1,3}\b',                       "IP address"),
        (r'(?i)\b(?:http|https|ftp)://[^\s"\'<>]+',            "URL"),
        (r'(?i)[a-z0-9][a-z0-9\-\.]{4,}\.(xyz|top|tk|ml|ga|cf|gq|pw|cc|io)\b', "Suspicious TLD"),
        (r'\b(?:25[0-5]|2[0-4]\d|[01]?\d\d?)(?:\.(?:25[0-5]|2[0-4]\d|[01]?\d\d?)){3}:\d{2,5}\b', "IP:Port"),
        (r'(?i)(?:cmd|powershell|wscript|cscript|mshta|regsvr32|rundll32)\.exe', "Living off land binary"),
    ]

    @staticmethod
    def shannon_entropy(data: bytes) -> float:
        if not data: return 0.0
        c = Counter(data); l = len(data)
        return -sum((v/l)*math.log2(v/l) for v in c.values())

    @classmethod
    def full_analysis(cls, data: bytes, pe: PEParser) -> Dict:
        r = {}
        r["overall_entropy"]  = round(cls.shannon_entropy(data), 3)
        r["entropy_verdict"]  = (
            "HIGH — packed/encrypted/shellcode likely" if r["overall_entropy"] > 7.0 else
            "MEDIUM — possible compression or encoding"  if r["overall_entropy"] > 6.0 else
            "LOW — mostly plaintext content"
        )
        # Shellcode signatures
        r["shellcode_signatures"] = []
        for sig, label in cls.SHELLCODE_SIGS:
            off = data.find(sig)
            if off != -1:
                r["shellcode_signatures"].append(f"{label} @ 0x{off:08X}")
        # Packer signatures
        r["packer_signatures"] = []
        for sig, label in cls.PACKER_SIGS:
            if sig in data:
                r["packer_signatures"].append(label)
        # EXITFUNC markers
        r["exitfunc_markers"] = [f"0x{m.start():08X}" for m in re.finditer(b"EXITFUNC", data)]
        # NOP sled
        nop = data.find(b"\x90" * 16)
        r["nop_sled"] = f"0x{nop:08X}" if nop != -1 else None
        # Encoder artifacts
        r["encoder_artifacts"] = [a for a in cls.ENCODER_ARTIFACTS if a.encode('latin-1') in data]
        # Suspicious sections
        r["suspicious_sections"] = [
            f"{s['name']}: entropy={s['entropy']} flags={s['flags']}"
            for s in pe.sections if s.get("suspicious")
        ]
        # Embedded PE
        r["embedded_pe_headers"] = [f"0x{m.start():08X}" for m in re.finditer(b"MZ", data[0x200:])][:5]
        # C2 / network IOCs from string extraction
        all_strings_text = data.decode('latin-1', errors='replace')
        r["c2_indicators"] = []
        for pattern, label in cls.C2_PATTERNS:
            for match in re.findall(pattern, all_strings_text)[:5]:
                r["c2_indicators"].append(f"[{label}] {match}")
        # File hashes
        r["md5"]    = hashlib.md5(data).hexdigest()
        r["sha1"]   = hashlib.sha1(data).hexdigest()
        r["sha256"] = hashlib.sha256(data).hexdigest()
        # Byte frequency anomaly
        if data:
            freq = Counter(data)
            top = freq.most_common(3)
            r["byte_freq_top"] = [(f"0x{b:02X}", c) for b, c in top]
            zero_pct = freq.get(0, 0) / len(data) * 100
            r["null_byte_pct"] = round(zero_pct, 2)
        # Rich header (linker metadata)
        rich_off = data.find(b"Rich")
        r["rich_header_present"] = rich_off != -1
        # Overlay (data after end of PE)
        last_end = max((s["raw_offset"] + s["raw_size"] for s in pe.sections if s["raw_size"]), default=0)
        r["overlay_size"] = max(0, len(data) - last_end)
        r["has_overlay"] = r["overlay_size"] > 512
        return r


# ══════════════════════════════════════════════════════════════════════════════
#  YARA-LIKE RULE ENGINE (static pattern matching)
# ══════════════════════════════════════════════════════════════════════════════
class YaraEngine:
    """Simple static rule matching engine mimicking YARA logic."""
    RULES = [
        {
            "name": "Cobalt_Strike_Beacon",
            "strings": [b"cobaltstrike", b"beacon", b"C2Profile", b"license_id", b"MZ\x90\x00"],
            "threshold": 2,
            "severity": "CRITICAL",
            "description": "Cobalt Strike Beacon artifacts detected",
        },
        {
            "name": "Msfvenom_Payload",
            "strings": [b"EXITFUNC", b"meterpreter", b"reverse_tcp", b"bind_tcp", b"metsrv"],
            "threshold": 1,
            "severity": "HIGH",
            "description": "Metasploit/msfvenom payload artifacts",
        },
        {
            "name": "Mimikatz",
            "strings": [b"mimikatz", b"sekurlsa", b"lsadump", b"wdigest", b"kerberos", b"NTLM"],
            "threshold": 2,
            "severity": "CRITICAL",
            "description": "Mimikatz credential dumper signatures",
        },
        {
            "name": "Ransomware_Crypto",
            "strings": [b"CryptEncrypt", b"CryptGenRandom", b"BCryptEncrypt", b".locked", b"YOUR_FILES", b"README.txt"],
            "threshold": 2,
            "severity": "CRITICAL",
            "description": "Ransomware file encryption patterns",
        },
        {
            "name": "Keylogger",
            "strings": [b"GetAsyncKeyState", b"SetWindowsHookEx", b"WH_KEYBOARD", b"keylog", b"keystrokes"],
            "threshold": 2,
            "severity": "HIGH",
            "description": "Keylogger behavioral signatures",
        },
        {
            "name": "RAT_Network",
            "strings": [b"socket", b"connect", b"recv", b"cmd.exe", b"shell", b"backdoor"],
            "threshold": 3,
            "severity": "HIGH",
            "description": "Remote Access Trojan network/shell patterns",
        },
        {
            "name": "Credential_Theft",
            "strings": [b"CredEnumerate", b"CryptUnprotectData", b"NlpGetPrimaryCredential", b"SAMDumpHashes"],
            "threshold": 1,
            "severity": "HIGH",
            "description": "Credential theft API patterns",
        },
        {
            "name": "Rootkit_DKOM",
            "strings": [b"NtQuerySystemInformation", b"DKOM", b"FLINK", b"BLINK", b"ActiveProcessLinks"],
            "threshold": 2,
            "severity": "CRITICAL",
            "description": "Direct Kernel Object Manipulation / rootkit patterns",
        },
        {
            "name": "Dropper",
            "strings": [b"URLDownloadToFile", b"WinExec", b"ShellExecute", b"CreateProcess", b"DropPath"],
            "threshold": 2,
            "severity": "HIGH",
            "description": "File dropper / downloader patterns",
        },
        {
            "name": "Anti_Analysis",
            "strings": [b"IsDebuggerPresent", b"CheckRemoteDebuggerPresent", b"VirtualBox", b"vmware", b"QEMU"],
            "threshold": 2,
            "severity": "MEDIUM",
            "description": "Anti-analysis / sandbox evasion patterns",
        },
        {
            "name": "Bootkit_MBR",
            "strings": [b"\x55\xAA", b"MBR", b"GRUB", b"bootmgr", b"NtLoadDriver"],
            "threshold": 2,
            "severity": "CRITICAL",
            "description": "Bootkit / MBR infection patterns",
        },
        {
            "name": "Worm_Propagation",
            "strings": [b"NetShareEnum", b"CopyFile", b"WNetEnumResource", b"FindFirstFile", b"AUTORUN.INF"],
            "threshold": 2,
            "severity": "HIGH",
            "description": "Network/USB worm propagation patterns",
        },
    ]

    @classmethod
    def scan(cls, data: bytes) -> List[Dict]:
        hits = []
        lower_data = data.lower()
        for rule in cls.RULES:
            matched = [s for s in rule["strings"] if s.lower() in lower_data]
            if len(matched) >= rule["threshold"]:
                hits.append({
                    "rule": rule["name"],
                    "severity": rule["severity"],
                    "description": rule["description"],
                    "matched_strings": [s.decode('latin-1', errors='replace') for s in matched],
                    "match_count": len(matched),
                })
        return hits


# ══════════════════════════════════════════════════════════════════════════════
#  BEHAVIORAL HEURISTICS
# ══════════════════════════════════════════════════════════════════════════════
class BehavioralHeuristics:
    """Simulated behavioral analysis from static artifacts."""
    BEHAVIORS = {
        "file_ops": {
            "apis": ["CreateFileA","CreateFileW","DeleteFileA","WriteFile","ReadFile","MoveFileA","CopyFileA","FindFirstFileA"],
            "label": "File System Operations",
        },
        "registry_ops": {
            "apis": ["RegOpenKeyExA","RegSetValueExA","RegCreateKeyExA","RegDeleteKeyA","RegQueryValueExA"],
            "label": "Registry Manipulation",
        },
        "process_ops": {
            "apis": ["CreateProcessA","OpenProcess","TerminateProcess","NtCreateProcess","CreateThread"],
            "label": "Process Manipulation",
        },
        "network_ops": {
            "apis": ["WSAStartup","connect","send","recv","HttpOpenRequest","InternetConnect","WinHttpOpen"],
            "label": "Network Communication",
        },
        "crypto_ops": {
            "apis": ["CryptEncrypt","CryptDecrypt","CryptGenRandom","BCryptEncrypt","BCryptGenRandom"],
            "label": "Cryptographic Operations",
        },
        "privilege_ops": {
            "apis": ["AdjustTokenPrivileges","OpenProcessToken","DuplicateToken","ImpersonateLoggedOnUser"],
            "label": "Privilege Escalation",
        },
        "memory_ops": {
            "apis": ["VirtualAlloc","VirtualProtect","VirtualAllocEx","HeapCreate","MapViewOfFile"],
            "label": "Suspicious Memory Operations",
        },
        "ui_ops": {
            "apis": ["BlockInput","GetForegroundWindow","ShowWindow","SetWindowsHookEx","GetAsyncKeyState"],
            "label": "User Interface Interaction",
        },
    }

    @classmethod
    def analyze(cls, imports: Dict[str,List[str]]) -> Dict:
        all_apis = set()
        for funcs in imports.values():
            all_apis.update(f.lower() for f in funcs)
        results = {}
        for key, info in cls.BEHAVIORS.items():
            matched = [a for a in info["apis"] if a.lower() in all_apis]
            if matched:
                results[key] = {"label": info["label"], "matched": matched, "count": len(matched)}
        return results


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN APP CLASS
# ══════════════════════════════════════════════════════════════════════════════
class SentinelXApp:
    def __init__(self, groq_api_key: str):
        self.chat_model = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="llama-3.3-70b-versatile",
            temperature=0.7, max_tokens=None
        )
        self._build_chains()

    def _build_chains(self):
        self.code_prompt = PromptTemplate(
            input_variables=["code_chunk"],
            template="""You are a senior malware analyst at a SOC. Analyze this code for malicious indicators.

Code:
{code_chunk}

Return ONLY valid JSON (no markdown, no backticks):
{{
    "threat_level": "HIGH|MEDIUM|LOW|CLEAN",
    "verdict": "One sentence verdict",
    "summary": ["key finding 1", "key finding 2"],
    "sections": {{
        "code_obfuscation_techniques": {{"findings": [], "description": ""}},
        "suspicious_api_calls":        {{"findings": [], "description": ""}},
        "anti_analysis_mechanisms":    {{"findings": [], "description": ""}},
        "network_communication":       {{"findings": [], "description": ""}},
        "file_system_operations":      {{"findings": [], "description": ""}},
        "payload_analysis":            {{"findings": [], "description": ""}},
        "persistence_mechanisms":      {{"findings": [], "description": ""}},
        "privilege_escalation":        {{"findings": [], "description": ""}}
    }}
}}""",
        )

        self.binary_prompt = PromptTemplate(
            input_variables=["strings_chunk","forensics_report","pe_report","injection_report","yara_report","behavioral_report"],
            template="""You are a senior malware reverse engineer and SOC analyst.

═══ PE STRUCTURE ═══
{pe_report}

═══ BINARY FORENSICS ═══
{forensics_report}

═══ INJECTION TECHNIQUES ═══
{injection_report}

═══ YARA RULE MATCHES ═══
{yara_report}

═══ BEHAVIORAL ANALYSIS ═══
{behavioral_report}

═══ EXTRACTED STRINGS (sample) ═══
{strings_chunk}

Cross-reference ALL sources. Identify malware family if possible.

Return ONLY valid JSON:
{{
    "threat_level": "CRITICAL|HIGH|MEDIUM|LOW|CLEAN",
    "verdict": "One clear sentence naming family/technique if identifiable",
    "malware_family": "Family name or Unknown",
    "confidence": "HIGH|MEDIUM|LOW",
    "summary": ["finding1", "finding2", "finding3"],
    "sections": {{
        "payload_identification": {{"findings": [], "description": ""}},
        "injection_techniques":   {{"findings": [], "description": ""}},
        "shellcode_indicators":   {{"findings": [], "description": ""}},
        "command_and_control":    {{"findings": [], "description": ""}},
        "anti_analysis":          {{"findings": [], "description": ""}},
        "persistence":            {{"findings": [], "description": ""}},
        "encoding_obfuscation":   {{"findings": [], "description": ""}},
        "privilege_escalation":   {{"findings": [], "description": ""}},
        "behavioral_profile":     {{"findings": [], "description": ""}},
        "ioc_summary":            {{"findings": [], "description": "IOCs: IPs, URLs, hashes, registry keys, mutexes"}}
    }}
}}""",
        )

        self.code_chain   = LLMChain(llm=self.chat_model, prompt=self.code_prompt,   verbose=False)
        self.binary_chain = LLMChain(llm=self.chat_model, prompt=self.binary_prompt, verbose=False)
        self.chat_memory  = ConversationBufferMemory()
        self.conversation = ConversationChain(llm=self.chat_model, memory=self.chat_memory, verbose=False)

    def _clean_json(self, s: str) -> str:
        # strip <think> blocks if present
        s = re.sub(r'<think>.*?</think>', '', s, flags=re.DOTALL)
        start = s.find('{'); end = s.rfind('}') + 1
        if start != -1 and end: s = s[start:end]
        return s.replace('```json','').replace('```','').strip()

    def _chunks(self, text: str, size: int = 10000) -> List[str]:
        return [text[i:i+size] for i in range(0, len(text), size)]

    def _error(self, msg: str, binary: bool = False) -> Dict:
        secs = (["payload_identification","injection_techniques","shellcode_indicators","command_and_control","anti_analysis","persistence","encoding_obfuscation","privilege_escalation","behavioral_profile","ioc_summary"] if binary else ["code_obfuscation_techniques","suspicious_api_calls","anti_analysis_mechanisms","network_communication","file_system_operations","payload_analysis","persistence_mechanisms","privilege_escalation"])
        return {"threat_level":"UNKNOWN","verdict":"Analysis failed","malware_family":"Unknown","confidence":"LOW","summary":[msg],"sections":{s:{"findings":["Error"],"description":"Analysis failed"} for s in secs},"errors":[msg]}

    def _combine(self, analyses: List[Dict]) -> Dict:
        if not analyses: return {"summary":[],"sections":{},"errors":[]}
        order = {"CRITICAL":4,"HIGH":3,"MEDIUM":2,"LOW":1,"CLEAN":0,"UNKNOWN":0}
        combined = {"summary":set(),"sections":{},"errors":[],"threat_level":"CLEAN","verdict":"","malware_family":"Unknown","confidence":"LOW","risk_score":0}
        for s in analyses[0].get("sections",{}):
            combined["sections"][s] = {"findings":set(),"description":""}
        for a in analyses:
            if "error" in a: combined["errors"].append(a["error"])
            combined["summary"].update(a.get("summary",[]))
            if order.get(a.get("threat_level","CLEAN"),0) > order.get(combined["threat_level"],0):
                combined["threat_level"] = a.get("threat_level","CLEAN")
                combined["verdict"]      = a.get("verdict","")
                combined["malware_family"] = a.get("malware_family","Unknown")
                combined["confidence"]   = a.get("confidence","LOW")
            for sec, c in a.get("sections",{}).items():
                if sec not in combined["sections"]:
                    combined["sections"][sec] = {"findings":set(),"description":""}
                combined["sections"][sec]["findings"].update(c.get("findings",[]))
                if c.get("description") and not combined["sections"][sec]["description"]:
                    combined["sections"][sec]["description"] = c["description"]
        result = {k:combined[k] for k in ["threat_level","verdict","malware_family","confidence","errors"]}
        result["summary"] = list(combined["summary"])
        result["sections"] = {}
        for sec, c in combined["sections"].items():
            findings = list(c["findings"])
            if len(findings) > 1 and "Error" in findings: findings.remove("Error")
            result["sections"][sec] = {"findings":findings,"description":c["description"] or "No significant findings."}
        return result

    def analyze_code(self, code: str) -> Dict:
        chunks = self._chunks(code)
        analyses = []; bar = st.progress(0); status = st.empty()
        for i, chunk in enumerate(chunks, 1):
            status.markdown(f"<span style='font-family:Share Tech Mono,monospace;font-size:11px;color:#00d4ff;'>ANALYZING CHUNK {i}/{len(chunks)}...</span>", unsafe_allow_html=True)
            try:
                raw = self.code_chain.predict(code_chunk=chunk)
                analyses.append(json.loads(self._clean_json(raw)))
            except Exception as e:
                analyses.append(self._error(str(e)))
            bar.progress(i / len(chunks))
        status.empty(); bar.empty()
        return self._combine(analyses)

    def extract_strings(self, data: bytes, min_len: int = 6) -> List[str]:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.bin') as tmp:
            tmp.write(data); path = tmp.name
        try:
            try:
                out = subprocess.run(['strings','-n',str(min_len),path],capture_output=True,text=True,check=True).stdout
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
        sa = st.empty()
        sa.markdown("<span style='font-family:Share Tech Mono,monospace;font-size:11px;color:#00d4ff;'>► STEP 1/6 — PARSING PE STRUCTURE...</span>", unsafe_allow_html=True)
        pe = PEParser(data)

        sa.markdown("<span style='font-family:Share Tech Mono,monospace;font-size:11px;color:#00d4ff;'>► STEP 2/6 — ENTROPY & SHELLCODE ANALYSIS...</span>", unsafe_allow_html=True)
        forensics = BinaryForensics.full_analysis(data, pe)

        sa.markdown("<span style='font-family:Share Tech Mono,monospace;font-size:11px;color:#00d4ff;'>► STEP 3/6 — INJECTION TECHNIQUE DETECTION...</span>", unsafe_allow_html=True)
        string_list = self.extract_strings(data)
        injection   = InjectionDetector.detect(pe.imports, string_list)

        sa.markdown("<span style='font-family:Share Tech Mono,monospace;font-size:11px;color:#00d4ff;'>► STEP 4/6 — YARA RULE MATCHING...</span>", unsafe_allow_html=True)
        yara_hits = YaraEngine.scan(data)

        sa.markdown("<span style='font-family:Share Tech Mono,monospace;font-size:11px;color:#00d4ff;'>► STEP 5/6 — BEHAVIORAL HEURISTICS...</span>", unsafe_allow_html=True)
        behavioral = BehavioralHeuristics.analyze(pe.imports)

        pe_text        = self._format_pe(pe)
        forensics_text = self._format_forensics(forensics)
        injection_text = self._format_injection(injection)
        yara_text      = self._format_yara(yara_hits)
        behavioral_text= self._format_behavioral(behavioral)

        sa.markdown("<span style='font-family:Share Tech Mono,monospace;font-size:11px;color:#ffaa00;'>► STEP 6/6 — AI VERDICT ENGINE...</span>", unsafe_allow_html=True)
        strings_blob = "\n".join(string_list)
        chunks = self._chunks(strings_blob)
        analyses = []; bar = st.progress(0); ps = st.empty()
        for i, chunk in enumerate(chunks, 1):
            ps.markdown(f"<span style='font-family:Share Tech Mono,monospace;font-size:11px;color:#ffaa00;'>AI ANALYZING CHUNK {i}/{len(chunks)}...</span>", unsafe_allow_html=True)
            try:
                raw = self.binary_chain.predict(
                    strings_chunk=chunk, forensics_report=forensics_text,
                    pe_report=pe_text, injection_report=injection_text,
                    yara_report=yara_text, behavioral_report=behavioral_text,
                )
                analyses.append(json.loads(self._clean_json(raw)))
            except Exception as e:
                analyses.append(self._error(str(e), binary=True))
            bar.progress(i / len(chunks))
        ps.empty(); bar.empty(); sa.empty()
        result = self._combine(analyses)
        result.update({
            "pe_parsed":pe.result,"forensics":forensics,"injection":injection,
            "yara_hits":yara_hits,"behavioral":behavioral,
            "pe_text":pe_text,"forensics_text":forensics_text,
            "injection_text":injection_text,"yara_text":yara_text,
            "behavioral_text":behavioral_text,
        })
        return result

    def _format_pe(self, pe: PEParser) -> str:
        if not pe.valid: return f"NOT A VALID PE FILE: {pe.result.get('error','Unknown')}"
        r = pe.result
        lines = [f"Arch={r.get('architecture')} Type={r.get('pe_type')} EP={r.get('entry_point')}",
                 f"ImageBase={r.get('image_base')} Subsystem={r.get('subsystem')}",
                 f"Timestamp={r.get('timestamp_human')}", f"Chars={','.join(r.get('characteristics',[]))}",
                 f"DLLChars={','.join(r.get('dll_characteristics',[]))}", "SECTIONS:"]
        for s in r.get("sections",[]):
            lines.append(f"  {s['name']} entropy={s['entropy']} flags={s['flags']} {'⚠SUSPICIOUS' if s.get('suspicious') else ''}")
        lines.append("IMPORTS:")
        for dll, funcs in r.get("imports",{}).items():
            lines.append(f"  [{dll}]: {','.join(funcs[:40])}")
        return "\n".join(lines)

    def _format_forensics(self, f: Dict) -> str:
        return "\n".join([
            f"Entropy={f['overall_entropy']} [{f['entropy_verdict']}]",
            f"MD5={f.get('md5','')} SHA256={f.get('sha256','')}",
            f"Shellcode={f.get('shellcode_signatures') or 'None'}",
            f"Packers={f.get('packer_signatures') or 'None'}",
            f"EXITFUNC={f.get('exitfunc_markers') or 'None'}",
            f"NOP_sled={f.get('nop_sled') or 'None'}",
            f"EncoderArtifacts={f.get('encoder_artifacts') or 'None'}",
            f"C2_IOCs={f.get('c2_indicators') or 'None'}",
            f"Overlay={f.get('overlay_size',0)} bytes HasOverlay={f.get('has_overlay',False)}",
            f"RichHeader={f.get('rich_header_present',False)}",
        ])

    def _format_injection(self, inj: Dict) -> str:
        lines = [f"RiskScore={inj['risk_score']}"]
        for t in inj["detected_techniques"]:
            lines.append(f"  [{t['severity']}][{t['confidence']}] {t['technique']}: {','.join(t['matched_apis'])}")
        for combo in inj["suspicious_api_combinations"]:
            lines.append(f"  COMBO: {combo}")
        for cat, apis in inj["evasion_capabilities"].items():
            lines.append(f"  EVASION/{cat}: {','.join(apis)}")
        return "\n".join(lines)

    def _format_yara(self, hits: List[Dict]) -> str:
        if not hits: return "No YARA rules matched."
        return "\n".join([f"[{h['severity']}] {h['rule']}: {h['description']} (matched: {','.join(h['matched_strings'][:4])})" for h in hits])

    def _format_behavioral(self, beh: Dict) -> str:
        if not beh: return "No behavioral signatures detected."
        return "\n".join([f"{v['label']}: {','.join(v['matched'][:6])}" for v in beh.values()])

    def create_report(self, results: Dict, title: str = "SentinelX Security Report") -> str:
        doc = Document()
        doc.add_heading(title, 0)
        tl = results.get("threat_level","UNKNOWN")
        doc.add_heading(f"Threat Level: {tl}", 1)
        if results.get("verdict"): doc.add_paragraph(results["verdict"])
        if results.get("malware_family","Unknown") != "Unknown":
            doc.add_paragraph(f"Malware Family: {results['malware_family']}")
        if results.get("yara_hits"):
            doc.add_heading("YARA Rule Matches", 1)
            for h in results["yara_hits"]:
                doc.add_paragraph(f"[{h['severity']}] {h['rule']}: {h['description']}")
        if results.get("behavioral"):
            doc.add_heading("Behavioral Analysis", 1)
            for key, v in results["behavioral"].items():
                doc.add_paragraph(f"{v['label']}: {', '.join(v['matched'])}")
        for k, text_key in [("PE Structure","pe_text"),("Binary Forensics","forensics_text"),
                             ("Injection Analysis","injection_text")]:
            if results.get(text_key):
                doc.add_heading(k, 1)
                doc.add_paragraph(results[text_key])
        doc.add_heading("Executive Summary", 1)
        for pt in results.get("summary",["No summary"]):
            doc.add_paragraph(f"• {pt}")
        for sec, content in results.get("sections",{}).items():
            doc.add_heading(sec.replace('_',' ').title(), 1)
            if content.get("description"): doc.add_paragraph(content["description"])
            for f in content.get("findings",[]):
                if f not in ("Error","None identified"):
                    doc.add_paragraph(f"• {f}")
        fname = f"SentinelX_report_{os.getpid()}.docx"
        doc.save(fname); return fname

    def get_chat_response(self, user_input: str, analysis_context: str = "") -> str:
        if analysis_context:
            full_input = (
                f"You are a SOC analyst and cybersecurity teacher helping a student understand "
                f"a malware analysis result. Here is the full analysis that was just performed:\n\n"
                f"{analysis_context}\n\n"
                f"---\n"
                f"Student's question: {user_input}\n\n"
                f"Answer clearly and educationally. Refer to specific findings from the analysis above "
                f"where relevant. Explain WHY something is dangerous, not just what it is. Be concise."
            )
        else:
            full_input = (
                f"You are a SOC analyst and cybersecurity teacher. "
                f"No analysis has been run yet — answer this general question concisely:\n\n{user_input}"
            )
        response = self.conversation.predict(input=full_input)
        return re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()

    @staticmethod
    def build_analysis_context(results: Dict) -> str:
        """Flatten the analysis results into a readable text block for the LLM."""
        lines = []
        lines.append(f"THREAT LEVEL: {results.get('threat_level','UNKNOWN')}")
        lines.append(f"VERDICT: {results.get('verdict','N/A')}")
        lines.append(f"MALWARE FAMILY: {results.get('malware_family','Unknown')}")
        lines.append(f"CONFIDENCE: {results.get('confidence','LOW')}")
        lines.append("")

        summary = results.get("summary", [])
        if summary:
            lines.append("SUMMARY:")
            for s in summary:
                lines.append(f"  - {s}")
            lines.append("")

        # YARA hits
        yara = results.get("yara_hits", [])
        if yara:
            lines.append("YARA RULE MATCHES:")
            for h in yara:
                lines.append(f"  [{h['severity']}] {h['rule']}: {h['description']}")
                lines.append(f"    Matched strings: {', '.join(h['matched_strings'][:6])}")
            lines.append("")

        # Injection techniques
        inj = results.get("injection", {})
        if inj:
            lines.append(f"INJECTION RISK SCORE: {inj.get('risk_score', 0)}")
            techs = inj.get("detected_techniques", [])
            if techs:
                lines.append("DETECTED INJECTION TECHNIQUES:")
                for t in techs:
                    lines.append(f"  [{t['severity']}][confidence:{t['confidence']}] {t['technique']}")
                    lines.append(f"    APIs: {', '.join(t['matched_apis'])}")
                    lines.append(f"    Description: {t['description']}")
            evasion = inj.get("evasion_capabilities", {})
            if evasion:
                lines.append("EVASION CAPABILITIES:")
                for cat, apis in evasion.items():
                    lines.append(f"  {cat}: {', '.join(apis)}")
            lines.append("")

        # Forensics
        forensics = results.get("forensics", {})
        if forensics:
            lines.append(f"ENTROPY: {forensics.get('overall_entropy','?')} — {forensics.get('entropy_verdict','')}")
            lines.append(f"MD5: {forensics.get('md5','N/A')}")
            lines.append(f"SHA256: {forensics.get('sha256','N/A')}")
            if forensics.get("shellcode_signatures"):
                lines.append(f"SHELLCODE SIGNATURES: {', '.join(forensics['shellcode_signatures'])}")
            if forensics.get("packer_signatures"):
                lines.append(f"PACKERS DETECTED: {', '.join(forensics['packer_signatures'])}")
            if forensics.get("c2_indicators"):
                lines.append(f"C2 IOCs: {', '.join(forensics['c2_indicators'])}")
            if forensics.get("encoder_artifacts"):
                lines.append(f"ENCODER ARTIFACTS: {', '.join(forensics['encoder_artifacts'])}")
            lines.append("")

        # Behavioral
        behavioral = results.get("behavioral", {})
        if behavioral:
            lines.append("BEHAVIORAL PROFILE:")
            for v in behavioral.values():
                lines.append(f"  {v['label']}: {', '.join(v['matched'][:8])}")
            lines.append("")

        # AI sections
        for sec, content in results.get("sections", {}).items():
            findings = [f for f in content.get("findings", []) if f not in ("Error", "None identified")]
            if findings:
                lines.append(f"{sec.replace('_',' ').upper()}:")
                for f in findings[:6]:
                    lines.append(f"  - {f}")
                lines.append("")

        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  DISPLAY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
def display_overview_dashboard(results: Dict):
    tl = results.get("threat_level","UNKNOWN")
    inj = results.get("injection",{})
    yara = results.get("yara_hits",[])
    beh  = results.get("behavioral",{})
    forensics = results.get("forensics",{})

    threat_badge(tl, inj.get("risk_score",0))

    # verdict + family
    if results.get("verdict"):
        st.markdown(f"""
        <div style="background:#080f17;border:1px solid #0d2137;padding:12px 16px;margin:8px 0;
                    font-family:'Share Tech Mono',monospace;font-size:12px;color:#c8e6f5;">
          <span style="color:#4a7a99;">VERDICT ▸ </span>{results["verdict"]}
          {"&nbsp;&nbsp;<span style='background:#ff3c6e22;color:#ff3c6e;padding:2px 8px;'>"+results["malware_family"]+"</span>" if results.get("malware_family","Unknown") != "Unknown" else ""}
        </div>
        """, unsafe_allow_html=True)

    # quick stats row
    colors_map = {"CRITICAL":"#ff3c6e","HIGH":"#ff6b35","MEDIUM":"#ffaa00","LOW":"#00ff88","CLEAN":"#00ff88","UNKNOWN":"#4a7a99"}
    stat_row([
        ("YARA MATCHES", str(len(yara)), "#ff3c6e" if yara else "#00ff88"),
        ("INJECT TECHNIQUES", str(len(inj.get("detected_techniques",[]))), "#ff3c6e" if inj.get("detected_techniques") else "#00ff88"),
        ("BEHAVIORAL SIGS", str(len(beh)), "#ffaa00" if beh else "#00ff88"),
        ("EVASION CAPS", str(len(inj.get("evasion_capabilities",{}))), "#ffaa00" if inj.get("evasion_capabilities") else "#00ff88"),
        ("ENTROPY", str(forensics.get("overall_entropy","?")), "#ff3c6e" if forensics.get("overall_entropy",0) > 7 else "#ffaa00" if forensics.get("overall_entropy",0) > 6 else "#00ff88"),
        ("C2 IOCS", str(len(forensics.get("c2_indicators",[]))), "#ff3c6e" if forensics.get("c2_indicators") else "#00ff88"),
    ])

def display_yara_section(results: Dict):
    yara_hits = results.get("yara_hits",[])
    panel_header("YARA RULE ENGINE", f"{len(yara_hits)} rules matched", "#ff3c6e" if yara_hits else "#00ff88")
    if not yara_hits:
        st.markdown("<span style='font-family:Share Tech Mono,monospace;font-size:11px;color:#00ff88;'>● NO YARA RULES MATCHED</span>", unsafe_allow_html=True)
        return
    for h in yara_hits:
        sev_color = {"CRITICAL":"#ff3c6e","HIGH":"#ff6b35","MEDIUM":"#ffaa00","LOW":"#00ff88"}.get(h["severity"],"#4a7a99")
        st.markdown(f"""
        <div style="background:#050a0f;border:1px solid #0d2137;border-left:3px solid {sev_color};
                    padding:10px 14px;margin:5px 0;">
          <div style="display:flex;align-items:center;gap:10px;">
            <span style="font-family:'Orbitron',sans-serif;font-size:11px;font-weight:700;color:{sev_color};">{h['rule']}</span>
            <span style="font-family:'Share Tech Mono',monospace;font-size:9px;background:{sev_color}22;
                         color:{sev_color};padding:2px 6px;">{h['severity']}</span>
            <span style="font-family:'Share Tech Mono',monospace;font-size:9px;color:#4a7a99;margin-left:auto;">
              {h['match_count']} strings matched
            </span>
          </div>
          <div style="font-family:'Share Tech Mono',monospace;font-size:10px;color:#4a7a99;margin-top:4px;">{h['description']}</div>
          <div style="font-family:'Share Tech Mono',monospace;font-size:10px;color:#00d4ff;margin-top:4px;">
            {' '.join(['<span style="background:#00d4ff11;padding:1px 4px;margin:1px;">'+s+'</span>' for s in h['matched_strings'][:6]])}
          </div>
        </div>
        """, unsafe_allow_html=True)

def display_behavioral_section(results: Dict):
    beh = results.get("behavioral",{})
    panel_header("BEHAVIORAL ANALYSIS", "Simulated from static artifacts", "#ffaa00")
    if not beh:
        st.markdown("<span style='font-family:Share Tech Mono,monospace;font-size:11px;color:#00ff88;'>● NO BEHAVIORAL SIGNATURES DETECTED</span>", unsafe_allow_html=True)
        return
    cols = st.columns(2)
    for i, (key, v) in enumerate(beh.items()):
        with cols[i % 2]:
            st.markdown(f"""
            <div style="background:#050a0f;border:1px solid #0d2137;border-top:2px solid #ffaa00;
                        padding:10px;margin:4px 0;">
              <div style="font-family:'Orbitron',sans-serif;font-size:10px;color:#ffaa00;
                          letter-spacing:1px;margin-bottom:6px;">{v['label']}</div>
              <div style="font-family:'Share Tech Mono',monospace;font-size:10px;color:#c8e6f5;">
                {' '.join(['<span style="background:#ffaa0011;color:#ffaa00;padding:1px 4px;margin:1px;">'+a+'</span>' for a in v['matched'][:6]])}
              </div>
            </div>
            """, unsafe_allow_html=True)

def display_forensics_section(results: Dict):
    f = results.get("forensics")
    if not f: return
    panel_header("BINARY FORENSICS", "Entropy · Signatures · IOCs · Hashes", "#00d4ff")
    entropy_bar(f.get("overall_entropy", 0))
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div style="background:#050a0f;border:1px solid #0d2137;padding:10px;font-family:'Share Tech Mono',monospace;font-size:10px;">
          <div style="color:#4a7a99;margin-bottom:4px;">FILE HASHES</div>
          <div style="color:#00ff88;">MD5 &nbsp; {f.get('md5','N/A')}</div>
          <div style="color:#00d4ff;">SHA1 &nbsp; {f.get('sha1','N/A')[:40]}...</div>
          <div style="color:#ffaa00;">SHA256 {f.get('sha256','N/A')[:40]}...</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        overlay = f.get("overlay_size", 0)
        st.markdown(f"""
        <div style="background:#050a0f;border:1px solid #0d2137;padding:10px;font-family:'Share Tech Mono',monospace;font-size:10px;">
          <div style="color:#4a7a99;margin-bottom:4px;">FILE METADATA</div>
          <div>Rich Header: <span style="color:{'#ff3c6e' if f.get('rich_header_present') else '#00ff88'};">{'PRESENT' if f.get('rich_header_present') else 'ABSENT'}</span></div>
          <div>Overlay: <span style="color:{'#ff3c6e' if f.get('has_overlay') else '#00ff88'};">{overlay} bytes</span></div>
          <div>Null Bytes: <span style="color:#c8e6f5;">{f.get('null_byte_pct',0)}%</span></div>
        </div>
        """, unsafe_allow_html=True)
    if f.get("shellcode_signatures"):
        panel_header("SHELLCODE SIGNATURES", "", "#ff3c6e")
        for s in f["shellcode_signatures"]: finding_card(s, "critical")
    if f.get("packer_signatures"):
        panel_header("PACKER SIGNATURES", "", "#ff3c6e")
        for s in f["packer_signatures"]: finding_card(s, "high")
    if f.get("c2_indicators"):
        panel_header("C2 / NETWORK IOCS", "", "#ff3c6e")
        for ioc in f["c2_indicators"]: finding_card(ioc, "high")
    for label, key, color in [
        ("EXITFUNC MARKERS", "exitfunc_markers", "high"),
        ("ENCODER ARTIFACTS", "encoder_artifacts", "medium"),
        ("EMBEDDED PE HEADERS", "embedded_pe_headers", "high"),
    ]:
        items = f.get(key, [])
        if items:
            panel_header(label, "", "#ffaa00")
            for item in (items if isinstance(items, list) else [items]):
                finding_card(str(item), color)

def display_injection_section(results: Dict):
    inj = results.get("injection")
    if not inj: return
    panel_header("CODE INJECTION DETECTION", f"Risk Score: {inj.get('risk_score',0)}", "#ff3c6e")
    techs = inj.get("detected_techniques",[])
    if techs:
        st.markdown(f"<div style='font-family:Share Tech Mono,monospace;font-size:11px;color:#ff3c6e;'>🚨 {len(techs)} INJECTION TECHNIQUE(S) DETECTED</div>", unsafe_allow_html=True)
        for t in techs:
            technique_card(t["technique"], t["severity"], t["confidence"], t["description"], t["matched_apis"])
    else:
        st.markdown("<span style='font-family:Share Tech Mono,monospace;font-size:11px;color:#00ff88;'>● NO INJECTION TECHNIQUES DETECTED IN IMPORT TABLE</span>", unsafe_allow_html=True)
    if inj.get("suspicious_api_combinations"):
        panel_header("SUSPICIOUS API COMBOS", "", "#ffaa00")
        for c in inj["suspicious_api_combinations"]: finding_card(c, "high")
    if inj.get("evasion_capabilities"):
        panel_header("EVASION CAPABILITIES", "", "#ffaa00")
        for cat, apis in inj["evasion_capabilities"].items():
            finding_card(f"[{cat.upper()}] {', '.join(apis)}", "medium")

def display_pe_section(results: Dict):
    pe = results.get("pe_parsed")
    if not pe or not pe.get("architecture"): return
    panel_header("PE STRUCTURE ANALYSIS", "", "#00d4ff")
    stat_row([
        ("ARCHITECTURE", pe.get("architecture","?"), "#00d4ff"),
        ("TYPE", pe.get("pe_type","?"), "#00d4ff"),
        ("ENTRY POINT", pe.get("entry_point","?"), "#ffaa00"),
        ("SUBSYSTEM", pe.get("subsystem","?"), "#00d4ff"),
        ("SECTIONS", str(pe.get("num_sections","?")), "#00d4ff"),
    ])
    st.markdown(f"<div style='font-family:Share Tech Mono,monospace;font-size:10px;color:#4a7a99;margin:8px 0;'>TIMESTAMP: {pe.get('timestamp_human','?')} | CHARS: {', '.join(pe.get('characteristics',[]))} | DLL: {', '.join(pe.get('dll_characteristics',[]))}</div>", unsafe_allow_html=True)
    panel_header("SECTION TABLE", "", "#00d4ff")
    for s in pe.get("sections",[]):
        badge = "🔴" if s.get("suspicious") else "🟢"
        color = "#ff3c6e" if s.get("suspicious") else "#00ff88"
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:12px;background:#050a0f;
                    border-left:2px solid {color};padding:6px 10px;margin:2px 0;
                    font-family:'Share Tech Mono',monospace;font-size:11px;">
          <span>{badge}</span>
          <span style="color:{color};width:100px;">{s['name']}</span>
          <span style="color:#4a7a99;">VA:</span><span style="color:#c8e6f5;">{s['virtual_address']:08X}</span>
          <span style="color:#4a7a99;margin-left:8px;">ENTROPY:</span><span style="color:{color};">{s['entropy']}</span>
          <span style="color:#4a7a99;margin-left:8px;">FLAGS:</span><span style="color:#c8e6f5;">{', '.join(s['flags'])}</span>
          {"<span style='color:#ff3c6e;margin-left:auto;font-size:9px;'>⚠ W+X</span>" if s.get("suspicious") else ""}
        </div>
        """, unsafe_allow_html=True)
    if pe.get("imports"):
        panel_header("IMPORT TABLE", f"{sum(len(v) for v in pe['imports'].values())} total APIs", "#00d4ff")
        for dll, funcs in pe["imports"].items():
            with st.expander(f"  {dll}  ({len(funcs)} functions)"):
                st.code(", ".join(funcs), language=None)

def display_ai_verdict(results: Dict):
    panel_header("AI ANALYSIS VERDICT", "LLM-powered cross-correlation", "#00ff88")
    st.markdown(f"""
    <div style="background:#050a0f;border:1px solid #0d2137;border-top:2px solid #00ff88;padding:16px;margin:8px 0;">
      <div style="font-family:'Orbitron',sans-serif;font-size:11px;color:#4a7a99;margin-bottom:8px;letter-spacing:2px;">EXECUTIVE SUMMARY</div>
      {"".join(['<div style="font-family:Share Tech Mono,monospace;font-size:11px;color:#c8e6f5;padding:3px 0;border-bottom:1px solid #0d2137;">▸ '+pt+'</div>' for pt in results.get('summary',['No findings'])])}
    </div>
    """, unsafe_allow_html=True)
    for sec_name, content in results.get("sections",{}).items():
        if not content.get("findings") or content["findings"] == ["Error"]: continue
        panel_header(sec_name.replace('_',' ').upper(), content.get("description",""), "#00d4ff")
        for finding in content["findings"]:
            if finding not in ("Error","None identified"):
                finding_card(finding, "info")

def display_analysis_results(results: Dict):
    tab_overview, tab_yara, tab_inject, tab_forensics, tab_pe, tab_ai = st.tabs([
        "⬡ OVERVIEW", "◈ YARA ENGINE", "⚡ INJECTION", "🔬 FORENSICS", "🗂 PE STRUCTURE", "🤖 AI VERDICT"
    ])
    with tab_overview:  display_overview_dashboard(results)
    with tab_yara:      display_yara_section(results)
    with tab_inject:    display_injection_section(results)
    with tab_forensics:
        display_forensics_section(results)
        display_behavioral_section(results)
    with tab_pe:        display_pe_section(results)
    with tab_ai:        display_ai_verdict(results)


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
def build_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="font-family:'Orbitron',sans-serif;font-size:16px;font-weight:900;
                    color:#00d4ff;letter-spacing:3px;padding:8px 0 4px;
                    text-shadow:0 0 12px rgba(0,212,255,0.4);">⬡ SentinelX</div>
        <div style="font-family:'Share Tech Mono',monospace;font-size:9px;color:#1a3a52;
                    letter-spacing:2px;margin-bottom:16px;">MALWARE ANALYSIS PLATFORM</div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='font-family:Share Tech Mono,monospace;font-size:10px;color:#4a7a99;margin-bottom:4px;letter-spacing:1px;'>▸ GROQ API KEY</div>", unsafe_allow_html=True)
        key = st.text_input("", type="password", placeholder="gsk_...", label_visibility="collapsed")
        if key:
            st.markdown("<span style='font-family:Share Tech Mono,monospace;font-size:10px;color:#00ff88;'>● AUTH OK</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span style='font-family:Share Tech Mono,monospace;font-size:10px;color:#ff3c6e;'>● NO KEY</span>", unsafe_allow_html=True)

        st.markdown("<hr style='border-color:#0d2137;margin:16px 0;'>", unsafe_allow_html=True)
        st.markdown("<div style='font-family:Share Tech Mono,monospace;font-size:10px;color:#4a7a99;letter-spacing:1px;margin-bottom:8px;'>▸ NAVIGATION</div>", unsafe_allow_html=True)

        nav_items = [
            ("⬡", "ANALYZER",   "Binary & source analysis"),
            ("◈", "INTEL CHAT", "GAMKERSGPT assistant"),
            ("△", "ABOUT",      "Platform overview"),
        ]
        if "nav" not in st.session_state:
            st.session_state.nav = "ANALYZER"
        for icon, name, desc in nav_items:
            active = st.session_state.nav == name
            if st.button(f"{icon}  {name}", key=f"nav_{name}", use_container_width=True):
                st.session_state.nav = name
                st.rerun()

        st.markdown("<hr style='border-color:#0d2137;margin:16px 0;'>", unsafe_allow_html=True)
        st.markdown("""
        <div style="font-family:'Share Tech Mono',monospace;font-size:9px;color:#1a3a52;line-height:2;">
          DETECTION MODULES<br>
          <span style="color:#00ff88;">●</span> PE Parser v2<br>
          <span style="color:#00ff88;">●</span> YARA Engine (12 rules)<br>
          <span style="color:#00ff88;">●</span> Injection Detector (16 techniques)<br>
          <span style="color:#00ff88;">●</span> Behavioral Heuristics<br>
          <span style="color:#00ff88;">●</span> Entropy Analysis<br>
          <span style="color:#00ff88;">●</span> Shellcode Signatures<br>
          <span style="color:#00ff88;">●</span> C2 IOC Extractor<br>
          <span style="color:#00ff88;">●</span> AI Verdict Engine
        </div>
        """, unsafe_allow_html=True)

    return key


# ══════════════════════════════════════════════════════════════════════════════
#  ABOUT PAGE
# ══════════════════════════════════════════════════════════════════════════════
def about_page():
    soc_header()
    st.markdown("""
    <div style="max-width:800px;">
      <div style="font-family:'Orbitron',sans-serif;font-size:32px;font-weight:900;
                  color:#00d4ff;letter-spacing:4px;text-shadow:0 0 30px rgba(0,212,255,0.4);
                  margin-bottom:8px;">SentinelX SOC</div>
      <div style="font-family:'Share Tech Mono',monospace;font-size:13px;color:#4a7a99;
                  margin-bottom:32px;">ADVANCED MALWARE ANALYSIS PLATFORM — GAMKERS EDITION</div>
    </div>
    """, unsafe_allow_html=True)
    stat_row([
        ("INJECTION TECHNIQUES", "16", "#ff3c6e"),
        ("YARA RULES", "12", "#ff3c6e"),
        ("DETECTION MODULES", "8", "#ffaa00"),
        ("EVASION CATEGORIES", "8", "#ffaa00"),
        ("SHELLCODE SIGS", "13", "#00d4ff"),
        ("PACKER SIGS", "12", "#00d4ff"),
    ])
    st.markdown("<br>", unsafe_allow_html=True)
    cols = st.columns(2)
    with cols[0]:
        panel_header("STATIC ANALYSIS MODULES", "", "#00d4ff")
        for item in ["PE Header Parser (full COFF/Optional/DataDir)","Section entropy analysis","Import/Export table parsing","Shellcode signature matching","Packer/protector detection","C2 IOC extraction (IPs, URLs, domains)","File hash generation (MD5/SHA1/SHA256)","Overlay & Rich header detection"]:
            finding_card(item, "info")
    with cols[1]:
        panel_header("DETECTION CAPABILITIES", "", "#ff3c6e")
        for item in ["Process Hollowing / Doppelgänging","DLL Injection (Classic, Reflective)","APC/Early Bird injection","Atom Bombing","Section injection","Thread execution hijacking","Heaven's Gate (WoW64 bypass)","Module Stomping / Phantom DLL","IAT/SetWindowsHookEx hooking","YARA family detection","Behavioral profiling from imports","Evasion: anti-debug, anti-VM, sandbox"]:
            finding_card(item, "high")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    groq_api_key = build_sidebar()
    nav = st.session_state.get("nav","ANALYZER")

    if nav == "ABOUT":
        about_page(); return

    soc_header()

    if not groq_api_key:
        st.markdown("""
        <div style="background:#080f17;border:1px solid #0d2137;border-left:3px solid #ffaa00;
                    padding:16px;font-family:'Share Tech Mono',monospace;font-size:12px;color:#ffaa00;">
          ▸ ENTER GROQ API KEY IN SIDEBAR TO ACTIVATE ANALYSIS ENGINE
        </div>
        """, unsafe_allow_html=True)
        return

    if "app" not in st.session_state or st.session_state.get("_api_key") != groq_api_key:
        st.session_state.app      = SentinelXApp(groq_api_key)
        st.session_state.messages = []
        st.session_state._api_key = groq_api_key

    # ── ANALYZER ──────────────────────────────────────────────────────────────
    if nav == "ANALYZER":
        tab_code, tab_src, tab_bin = st.tabs(["📝 PASTE CODE", "📁 SOURCE FILE", "💾 BINARY"])

        with tab_code:
            code_input = st.text_area("", height=280, placeholder="// Paste source code here for AI-powered malware analysis...", label_visibility="collapsed")
            if st.button("⬡  RUN ANALYSIS", key="analyze_pasted") and code_input:
                with st.spinner(""):
                    results = st.session_state.app.analyze_code(code_input)
                    st.session_state.last_analysis = results
                    st.session_state.last_analysis_name = "Pasted Code"
                    display_analysis_results(results)
                    fname = st.session_state.app.create_report(results)
                    with open(fname,"rb") as f:
                        st.download_button("📥 DOWNLOAD REPORT", data=f, file_name=fname,
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
                    os.remove(fname)

        with tab_src:
            uf = st.file_uploader("", type=['py','js','java','cpp','cs','php','rb','go','rs','ts'], key="src_file", label_visibility="collapsed")
            if st.button("⬡  ANALYZE SOURCE", key="analyze_src") and uf:
                with st.spinner(""):
                    results = st.session_state.app.analyze_code(uf.read().decode(errors='replace'))
                    st.session_state.last_analysis = results
                    st.session_state.last_analysis_name = uf.name
                    display_analysis_results(results)
                    fname = st.session_state.app.create_report(results)
                    with open(fname,"rb") as f:
                        st.download_button("📥 DOWNLOAD REPORT", data=f, file_name=fname,
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
                    os.remove(fname)

        with tab_bin:
            st.markdown("<div style='font-family:Share Tech Mono,monospace;font-size:11px;color:#4a7a99;margin-bottom:8px;'>UPLOAD PE BINARY — EXE / DLL / BIN — FULL STATIC ANALYSIS</div>", unsafe_allow_html=True)
            ub = st.file_uploader("", type=['exe','dll','bin','sys','drv'], key="bin_file", label_visibility="collapsed")
            if ub:
                st.markdown(f"<div style='font-family:Share Tech Mono,monospace;font-size:10px;color:#00d4ff;'>◈ LOADED: {ub.name} ({ub.size:,} bytes)</div>", unsafe_allow_html=True)
                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("⬡  ANALYZE BINARY", key="analyze_bin"):
                        binary_data = ub.read()
                        with st.expander("▸ EXTRACTED STRINGS PREVIEW"):
                            preview = st.session_state.app.extract_strings(binary_data)
                            st.text_area("", value="\n".join(preview[:300]), height=200,
                                         disabled=True, label_visibility="collapsed")
                        results = st.session_state.app.analyze_binary(binary_data)
                        st.session_state.last_analysis = results
                        st.session_state.last_analysis_name = ub.name
                        display_analysis_results(results)
                        fname = st.session_state.app.create_report(results, title=f"Binary Analysis — {ub.name}")
                        with open(fname,"rb") as f:
                            st.download_button("📥 DOWNLOAD BINARY REPORT", data=f, file_name=fname,
                                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
                        os.remove(fname)

    # ── INTEL CHAT ────────────────────────────────────────────────────────────
    elif nav == "INTEL CHAT":
        st.markdown("""
        <div style="font-family:'Orbitron',sans-serif;font-size:13px;color:#00d4ff;
                    letter-spacing:2px;margin-bottom:4px;">GAMKERSGPT</div>
        <div style="font-family:'Share Tech Mono',monospace;font-size:11px;color:#4a7a99;margin-bottom:16px;">
          SOC INTELLIGENCE ASSISTANT — Ask about malware, injection techniques, IOC analysis, TTPs
        </div>
        """, unsafe_allow_html=True)

        # ── analysis context banner ──────────────────────────────────────────
        last = st.session_state.get("last_analysis")
        last_name = st.session_state.get("last_analysis_name", "Unknown")
        analysis_context = ""
        if last:
            tl = last.get("threat_level","UNKNOWN")
            tl_color = {"CRITICAL":"#ff3c6e","HIGH":"#ff6b35","MEDIUM":"#ffaa00","LOW":"#00ff88","CLEAN":"#00ff88"}.get(tl,"#4a7a99")
            st.markdown(f"""
            <div style="background:#050a0f;border:1px solid #0d2137;border-left:3px solid {tl_color};
                        padding:10px 14px;margin-bottom:12px;font-family:'Share Tech Mono',monospace;font-size:11px;">
              <span style="color:{tl_color};">◈ ANALYSIS CONTEXT LOADED</span>
              <span style="color:#4a7a99;margin:0 8px;">|</span>
              <span style="color:#c8e6f5;">{last_name}</span>
              <span style="color:#4a7a99;margin:0 8px;">|</span>
              <span style="color:{tl_color};">{tl}</span>
              <span style="color:#4a7a99;margin-left:16px;font-size:10px;">
                Ask anything about this analysis — the AI has full context of all findings
              </span>
            </div>
            """, unsafe_allow_html=True)
            analysis_context = SentinelXApp.build_analysis_context(last)

            col_clear, _ = st.columns([1, 5])
            with col_clear:
                if st.button("✕  CLEAR CONTEXT", key="clear_ctx"):
                    st.session_state.last_analysis = None
                    st.session_state.last_analysis_name = None
                    st.session_state.messages = []
                    st.rerun()
        else:
            st.markdown("""
            <div style="background:#050a0f;border:1px solid #0d2137;border-left:3px solid #1a3a52;
                        padding:10px 14px;margin-bottom:12px;font-family:'Share Tech Mono',monospace;font-size:11px;color:#4a7a99;">
              ◇ NO ANALYSIS LOADED — Run an analysis in the Analyzer tab first,
              then come back here to ask questions about the results.
              General cybersecurity questions still work.
            </div>
            """, unsafe_allow_html=True)

        # ── suggested questions when context is loaded ───────────────────────
        if last and not st.session_state.messages:
            st.markdown("<div style='font-family:Share Tech Mono,monospace;font-size:10px;color:#4a7a99;margin-bottom:6px;'>▸ SUGGESTED QUESTIONS</div>", unsafe_allow_html=True)
            suggestions = [
                "Why is this file considered malicious?",
                "Explain the injection technique detected in simple terms",
                "What does high entropy mean in this file?",
                "How would this malware persist on a system?",
                "What are the C2 indicators and what do they mean?",
                "How can I detect this malware on a live system?",
            ]
            cols = st.columns(3)
            for i, suggestion in enumerate(suggestions):
                with cols[i % 3]:
                    if st.button(suggestion, key=f"sug_{i}", use_container_width=True):
                        st.session_state.messages.append({"role":"user","content":suggestion})
                        with st.spinner(""):
                            resp = st.session_state.app.get_chat_response(suggestion, analysis_context)
                            st.session_state.messages.append({"role":"assistant","content":resp})
                        st.rerun()

        st.markdown("<hr style='border-color:#0d2137;margin:12px 0;'>", unsafe_allow_html=True)

        # ── chat history ─────────────────────────────────────────────────────
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("Ask about this analysis or any cybersecurity topic..."):
            st.session_state.messages.append({"role":"user","content":prompt})
            with st.chat_message("user"): st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner(""):
                    response = st.session_state.app.get_chat_response(prompt, analysis_context)
                    st.markdown(response)
                    st.session_state.messages.append({"role":"assistant","content":response})

if __name__ == "__main__":
    main()
