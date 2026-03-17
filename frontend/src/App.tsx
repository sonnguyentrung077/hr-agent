import { useState, useRef, useCallback, useEffect } from "react";

const WS_URL = "ws://localhost:8000/ws";
const TTS_SAMPLE_RATE = 44100;

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
  const [listening, setListening] = useState(false);
  const [partial, setPartial] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [status, setStatus] = useState("idle");
  const [resTime, setResTime] = useState<number | null>(null);

  const [streamingText, setStreamingText] = useState("");

  const wsRef = useRef<WebSocket | null>(null);
  const recCtxRef = useRef<AudioContext | null>(null);
  const playCtxRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const nextPlayRef = useRef(0);
  const bottomRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, partial, streamingText]);

  /* ---- playback: Cartesia sends pcm_f32le ---- */
  const queueAudio = useCallback((pcmBuf: ArrayBuffer) => {
    const ctx = playCtxRef.current;
    if (!ctx) return;
    const f32 = new Float32Array(pcmBuf);
    const buf = ctx.createBuffer(1, f32.length, TTS_SAMPLE_RATE);
    buf.getChannelData(0).set(f32);
    const src = ctx.createBufferSource();
    src.buffer = buf;
    src.connect(ctx.destination);
    const now = ctx.currentTime;
    const t = Math.max(now + 0.02, nextPlayRef.current);
    src.start(t);
    nextPlayRef.current = t + buf.duration;
  }, []);

  /* ---- WS message handler ---- */
  const onWsMessage = useCallback(
    (ev: MessageEvent) => {
      if (ev.data instanceof ArrayBuffer) {
        queueAudio(ev.data);
        return;
      }
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
          if (msg.status === "thinking") {
            setStreamingText("");
          }
          setStatus(msg.status === "thinking" ? "thinking" : "listening");
          break;
        case "response_text":
          setStreamingText((prev) => prev + (msg.token ?? ""));
          break;
        case "response_complete":
          setStreamingText("");
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
        default:
          break;
      }
    },
    [queueAudio],
  );

  /* ---- start ---- */
  const start = useCallback(async () => {
    try {
      playCtxRef.current = new AudioContext({ sampleRate: TTS_SAMPLE_RATE });
      nextPlayRef.current = 0;
      recCtxRef.current = new AudioContext({ sampleRate: 16000 });

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, sampleRate: 16000 },
      });
      streamRef.current = stream;

      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;
      ws.binaryType = "arraybuffer";

      ws.onopen = async () => {
        setStatus("listening");
        setListening(true);
        const ctx = recCtxRef.current!;
        await ctx.audioWorklet.addModule("/audio-processor.js");
        const source = ctx.createMediaStreamSource(stream);
        const node = new AudioWorkletNode(ctx, "audio-processor");
        node.port.onmessage = (e: MessageEvent<ArrayBuffer>) => {
          if (ws.readyState === WebSocket.OPEN) ws.send(e.data);
        };
        source.connect(node);
      };

      ws.onmessage = onWsMessage;
      ws.onerror = () => setStatus("error");
      ws.onclose = () => {
        setStatus("idle");
        setListening(false);
      };
    } catch (err) {
      console.error(err);
      setStatus("error");
    }
  }, [onWsMessage]);

  /* ---- stop ---- */
  const stop = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: "end_session" }));
      wsRef.current.close();
    }
    streamRef.current?.getTracks().forEach((t) => t.stop());
    recCtxRef.current?.close().catch(() => {});
    playCtxRef.current?.close().catch(() => {});
    setListening(false);
    setPartial("");
    setStatus("idle");
  }, []);

  const statusLabel =
    status === "listening"
      ? "Listening..."
      : status === "thinking"
        ? "Thinking..."
        : status === "error"
          ? "Connection error"
          : "";

  return (
    <div className="app">
      {/* ---- Hero / Mic area ---- */}
      <div className="hero">
        <button
          className={`mic-btn ${listening ? "active" : ""} ${status === "thinking" ? "thinking" : ""}`}
          onClick={listening ? stop : start}
          aria-label={listening ? "Stop" : "Start"}
        >
          {/* mic svg */}
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.8} strokeLinecap="round" strokeLinejoin="round">
            <rect x="9" y="2" width="6" height="11" rx="3" />
            <path d="M5 10a7 7 0 0 0 14 0" />
            <line x1="12" y1="17" x2="12" y2="22" />
            <line x1="8" y1="22" x2="16" y2="22" />
          </svg>
          {listening && <span className="pulse-ring" />}
          {listening && <span className="pulse-ring delay" />}
        </button>

        <div className="hero-text">
          <h1>Voice Assistant</h1>
          {!listening && !statusLabel && (
            <p className="hero-sub">Tap the mic to start talking</p>
          )}
          {statusLabel && (
            <p className={`hero-status ${status}`}>
              <span className="dot" />
              {statusLabel}
              {resTime != null && status === "listening" && (
                <span className="res-time">{resTime}s</span>
              )}
            </p>
          )}
        </div>
      </div>

      {/* ---- Chat area ---- */}
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
            <div className="bubble-name">{m.role === "user" ? "You" : "Assistant"}</div>
            <p>{m.text}</p>
          </div>
        ))}

        {streamingText && (
          <div className="bubble assistant streaming-bubble">
            <div className="bubble-name">Assistant</div>
            <p>{streamingText}<span className="cursor" /></p>
          </div>
        )}

        {partial && (
          <div className="bubble user partial-bubble">
            <div className="bubble-name">You</div>
            <p>{partial}</p>
          </div>
        )}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}
