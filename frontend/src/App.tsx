import { useState, useRef, useCallback } from "react";

const WS_URL = "ws://localhost:8000/ws";
const TTS_SAMPLE_RATE = 44100;

interface Message {
  role: "user" | "assistant";
  text: string;
}

interface WsMsg {
  type: string;
  text?: string;
  status?: string;
  message?: string;
  time?: number;
}

export default function App() {
  const [listening, setListening] = useState(false);
  const [partial, setPartial] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [status, setStatus] = useState("Click Start to begin");
  const [resTime, setResTime] = useState<number | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const recCtxRef = useRef<AudioContext | null>(null);
  const playCtxRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const nextPlayRef = useRef(0);

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
          setStatus(msg.status === "thinking" ? "Thinking..." : "Listening...");
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
          setStatus("Error: " + (msg.message ?? "unknown"));
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
        setStatus("Listening...");
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
      ws.onerror = () => setStatus("Connection error");
      ws.onclose = () => {
        setStatus("Disconnected");
        setListening(false);
      };
    } catch (err) {
      setStatus("Error: " + (err as Error).message);
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
    setStatus("Click Start to begin");
  }, []);

  return (
    <div className="app">
      <h1>Voice Assistant</h1>

      <div className="controls">
        <button
          className={`mic-btn ${listening ? "active" : ""}`}
          onClick={listening ? stop : start}
        >
          {listening ? "Stop" : "Start"}
        </button>
        <span className="status">{status}</span>
        {resTime != null && <span className="time">Response: {resTime}s</span>}
      </div>

      {partial && <div className="partial">{partial}</div>}

      <div className="messages">
        {messages.map((m, i) => (
          <div key={i} className={`msg ${m.role}`}>
            <strong>{m.role === "user" ? "You" : "Assistant"}</strong>
            <p>{m.text}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
