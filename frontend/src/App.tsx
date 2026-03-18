import { useState, useRef, useCallback, useEffect } from "react";

const BACKEND = "";
const WS_BASE = `ws://${window.location.host}`;

interface Message {
  role: "user" | "assistant";
  text: string;
}

interface WsMsg {
  type: string;
  text?: string;
  token?: string;
  status?: string;
  message?: string;
  time?: number;
}

export default function App() {
  const [sessionActive, setSessionActive] = useState(false);
  const [micOpen, setMicOpen] = useState(false);
  const [micLocked, setMicLocked] = useState(false); // persistent always-on toggle
  const [partial, setPartial] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [status, setStatus] = useState("idle");
  const [resTime, setResTime] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);

  const videoRef = useRef<HTMLVideoElement>(null);
  const audioRef = useRef<HTMLAudioElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const pcRef = useRef<RTCPeerConnection | null>(null);
  const recCtxRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const workletNodeRef = useRef<AudioWorkletNode | null>(null);
  const micSourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const micReadyRef = useRef(false);

  /* ---- WS message handler ---- */
  const onWsMessage = useCallback((ev: MessageEvent) => {
    if (ev.data instanceof ArrayBuffer) return;
    const msg: WsMsg = JSON.parse(ev.data as string);
    switch (msg.type) {
      case "partial_transcript":
        setPartial(msg.text ?? "");
        break;
      case "final_transcript":
        setPartial("");
        setMessages((p) => [...p, { role: "user", text: msg.text ?? "" }]);
        break;
      case "status":
        setStatus(msg.status === "thinking" ? "thinking" : "connected");
        break;
      case "response_complete":
        setMessages((p) => [
          ...p,
          { role: "assistant", text: msg.text ?? "" },
        ]);
        setResTime(msg.time ?? null);
        break;
      case "error":
      case "tts_error":
        setStatus("error");
        break;
    }
  }, []);

  /* ---- Start session: WebRTC + WebSocket (no mic yet) ---- */
  const startSession = useCallback(async () => {
    if (loading) return;
    setLoading(true);

    if (wsRef.current) {
      wsRef.current.onclose = null;
      if (wsRef.current.readyState === WebSocket.OPEN) wsRef.current.close();
      wsRef.current = null;
    }
    if (pcRef.current) { pcRef.current.close(); pcRef.current = null; }
    closeMicRefs();

    try {
      setStatus("Connecting...");

      const pc = new RTCPeerConnection({
        iceServers: [{ urls: "stun:stun.l.google.com:19302" }],
      });
      pcRef.current = pc;

      pc.ontrack = (ev) => {
        if (ev.track.kind === "video" && videoRef.current) {
          videoRef.current.srcObject = new MediaStream([ev.track]);
        } else if (ev.track.kind === "audio" && audioRef.current) {
          audioRef.current.srcObject = new MediaStream([ev.track]);
        }
      };

      pc.addTransceiver("video", { direction: "recvonly" });
      pc.addTransceiver("audio", { direction: "recvonly" });

      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);

      await new Promise<void>((resolve) => {
        if (pc.iceGatheringState === "complete") return resolve();
        const timeout = setTimeout(resolve, 2000);
        pc.onicegatheringstatechange = () => {
          if (pc.iceGatheringState === "complete") {
            clearTimeout(timeout);
            resolve();
          }
        };
      });

      const resp = await fetch(`${BACKEND}/offer`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          sdp: pc.localDescription!.sdp,
          type: pc.localDescription!.type,
        }),
      });
      const answer = await resp.json();
      await pc.setRemoteDescription(
        new RTCSessionDescription({ sdp: answer.sdp, type: answer.type })
      );
      const sessionId = answer.session_id;

      const ws = new WebSocket(`${WS_BASE}/ws?session_id=${sessionId}`);
      wsRef.current = ws;
      ws.binaryType = "arraybuffer";

      ws.onopen = async () => {
        setStatus("connected");
        setSessionActive(true);
        setLoading(false);

        // Pre-init mic: get permission + load worklet once
        try {
          const recCtx = new AudioContext({ sampleRate: 16000 });
          recCtxRef.current = recCtx;
          const micStream = await navigator.mediaDevices.getUserMedia({
            audio: { channelCount: 1, sampleRate: 16000 },
          });
          streamRef.current = micStream;
          await recCtx.audioWorklet.addModule("/audio-processor.js");
          const source = recCtx.createMediaStreamSource(micStream);
          micSourceRef.current = source;
          const node = new AudioWorkletNode(recCtx, "audio-processor");
          workletNodeRef.current = node;
          node.port.onmessage = (e: MessageEvent<ArrayBuffer>) => {
            if (ws.readyState === WebSocket.OPEN) ws.send(e.data);
          };
          // Don't connect yet — mic starts muted
          micReadyRef.current = true;
        } catch (err) {
          console.error("Mic init failed:", err);
        }
      };

      ws.onmessage = onWsMessage;
      ws.onerror = () => { setStatus("error"); setLoading(false); };
      ws.onclose = () => {
        setStatus("idle");
        setSessionActive(false);
        setMicOpen(false);
        setMicLocked(false);
        setLoading(false);
      };
    } catch (err) {
      console.error(err);
      setStatus("error");
      setLoading(false);
    }
  }, [loading, onWsMessage]);

  /* ---- Stop session ---- */
  const stopSession = useCallback(() => {
    closeMicRefs();
    setMicOpen(false);
    setMicLocked(false);

    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: "end_session" }));
      wsRef.current.close();
    }
    wsRef.current = null;

    pcRef.current?.close();
    pcRef.current = null;

    if (videoRef.current) videoRef.current.srcObject = null;
    if (audioRef.current) audioRef.current.srcObject = null;

    setSessionActive(false);
    setPartial("");
    setStatus("idle");
  }, []);

  /* ---- Close session on page reload/close ---- */
  useEffect(() => {
    const cleanup = () => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ type: "end_session" }));
        wsRef.current.close();
      }
      pcRef.current?.close();
      streamRef.current?.getTracks().forEach((t) => t.stop());
    };
    window.addEventListener("beforeunload", cleanup);
    return () => {
      window.removeEventListener("beforeunload", cleanup);
      cleanup();
    };
  }, []);

  /* ---- Mic open/close (connect/disconnect pre-initialized worklet) ---- */
  function closeMicRefs() {
    micSourceRef.current?.disconnect();
    micSourceRef.current = null;
    workletNodeRef.current?.disconnect();
    workletNodeRef.current = null;
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    recCtxRef.current?.close().catch(() => {});
    recCtxRef.current = null;
    micReadyRef.current = false;
  }

  const openMic = useCallback(() => {
    if (!micReadyRef.current || !micSourceRef.current || !workletNodeRef.current) return;
    micSourceRef.current.connect(workletNodeRef.current);
    setMicOpen(true);
  }, []);

  const closeMic = useCallback(() => {
    if (micSourceRef.current) {
      try { micSourceRef.current.disconnect(); } catch {}
    }
    setMicOpen(false);
  }, []);

  /* ---- Persistent toggle (small button) ---- */
  const toggleMicLock = useCallback(() => {
    if (micLocked) {
      // Turning off persistent mic
      setMicLocked(false);
      closeMic();
    } else {
      // Turning on persistent mic
      setMicLocked(true);
      openMic();
    }
  }, [micLocked, closeMic, openMic]);

  /* ---- Push-to-talk: big button + spacebar ---- */
  const onPttDown = useCallback(() => {
    if (!micLocked) openMic();
  }, [micLocked, openMic]);

  const onPttUp = useCallback(() => {
    if (!micLocked) closeMic();
  }, [micLocked, closeMic]);

  /* ---- Spacebar push-to-talk ---- */
  const micLockedRef = useRef(micLocked);
  micLockedRef.current = micLocked;
  const sessionActiveRef = useRef(sessionActive);
  sessionActiveRef.current = sessionActive;
  const openMicRef = useRef(openMic);
  openMicRef.current = openMic;
  const closeMicRef = useRef(closeMic);
  closeMicRef.current = closeMic;

  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      if (e.code === "Space" && !e.repeat && !micLockedRef.current && sessionActiveRef.current) {
        e.preventDefault();
        openMicRef.current();
      }
    };
    const up = (e: KeyboardEvent) => {
      if (e.code === "Space" && !micLockedRef.current && sessionActiveRef.current) {
        e.preventDefault();
        closeMicRef.current();
      }
    };
    window.addEventListener("keydown", down);
    window.addEventListener("keyup", up);
    return () => {
      window.removeEventListener("keydown", down);
      window.removeEventListener("keyup", up);
    };
  }, []);

  const statusLabel =
    status === "thinking"
      ? "Thinking..."
      : status === "error"
        ? "Connection error"
        : micOpen
          ? "Listening..."
          : "";

  return (
    <div className="app">
      <h1>HR Interview Assistant</h1>

      <div className="split">
        {/* ---- Left: Avatar + controls ---- */}
        <div className="left-panel">
          <div className="avatar-container">
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="avatar-video"
            />
            <audio ref={audioRef} autoPlay />
            {!sessionActive && (
              <div className="avatar-placeholder">Avatar will appear here</div>
            )}
          </div>

          <div className="controls">
            <button
              className={`session-btn ${sessionActive ? "active" : ""}`}
              onClick={sessionActive ? stopSession : startSession}
              disabled={loading}
            >
              {loading ? "Connecting..." : sessionActive ? "End Session" : "Start Session"}
            </button>

            {sessionActive && (
              <div className="mic-controls">
                {/* Big button: push-to-talk (hold) */}
                <button
                  className={`mic-btn ${micOpen ? "active" : ""} ${status === "thinking" ? "thinking" : ""}`}
                  onMouseDown={onPttDown}
                  onMouseUp={onPttUp}
                  onMouseLeave={!micLocked && micOpen ? closeMic : undefined}
                  onTouchStart={onPttDown}
                  onTouchEnd={onPttUp}
                  aria-label="Push to talk"
                >
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.8} strokeLinecap="round" strokeLinejoin="round">
                    <rect x="9" y="2" width="6" height="11" rx="3" />
                    <path d="M5 10a7 7 0 0 0 14 0" />
                    <line x1="12" y1="17" x2="12" y2="22" />
                    <line x1="8" y1="22" x2="16" y2="22" />
                  </svg>
                  {micOpen && <span className="pulse-ring" />}
                  {micOpen && <span className="pulse-ring delay" />}
                </button>

                <span className="mic-hint">
                  {micLocked ? "Mic always on" : "Hold to talk \u00b7 Space"}
                </span>

                {/* Small toggle: persistent mic on/off */}
                <button
                  className={`mic-lock-btn ${micLocked ? "locked" : ""}`}
                  onClick={toggleMicLock}
                  aria-label={micLocked ? "Turn off mic" : "Keep mic on"}
                  title={micLocked ? "Mic is always on \u2014 click to turn off" : "Click to keep mic always on"}
                >
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
                    {micLocked ? (
                      <>
                        <rect x="9" y="2" width="6" height="11" rx="3" />
                        <path d="M5 10a7 7 0 0 0 14 0" />
                        <line x1="12" y1="17" x2="12" y2="22" />
                      </>
                    ) : (
                      <>
                        <line x1="1" y1="1" x2="23" y2="23" />
                        <path d="M9 9v3a3 3 0 0 0 5.12 2.12M15 9.34V4a3 3 0 0 0-5.94-.6" />
                        <path d="M17 16.95A7 7 0 0 1 5 12" />
                        <line x1="12" y1="19" x2="12" y2="22" />
                      </>
                    )}
                  </svg>
                </button>
              </div>
            )}

            {statusLabel && (
              <p className={`hero-status ${status}`}>
                <span className="dot" />
                {statusLabel}
                {resTime != null && (
                  <span className="res-time">{resTime}s</span>
                )}
              </p>
            )}
          </div>
        </div>

        {/* ---- Right: Chat log ---- */}
        <div className="chat">
          {messages.length === 0 && !partial && (
            <div className="empty">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.2} strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
              </svg>
              <span>Your conversation will appear here</span>
            </div>
          )}

          {messages.map((m, i) => (
            <div key={i} className={`bubble ${m.role}`}>
              <div className="bubble-name">{m.role === "user" ? "You" : "Interviewer"}</div>
              <p>{m.text}</p>
            </div>
          ))}

          {partial && (
            <div className="bubble user partial-bubble">
              <div className="bubble-name">You</div>
              <p>{partial}</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
