class AudioProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.targetRate = 16000;
    this.ratio = sampleRate / this.targetRate;
    this.buffer = [];
    this.chunkSize = 2048;
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0]) return true;

    const ch = input[0];

    // Downsample to 16 kHz
    for (let i = 0; i < ch.length; i += this.ratio) {
      const idx = Math.round(i);
      if (idx < ch.length) {
        this.buffer.push(ch[idx]);
      }
    }

    // Emit PCM16 chunks
    while (this.buffer.length >= this.chunkSize) {
      const slice = this.buffer.splice(0, this.chunkSize);
      const pcm16 = new Int16Array(slice.length);
      for (let j = 0; j < slice.length; j++) {
        const s = Math.max(-1, Math.min(1, slice[j]));
        pcm16[j] = s < 0 ? s * 0x8000 : s * 0x7fff;
      }
      this.port.postMessage(pcm16.buffer, [pcm16.buffer]);
    }

    return true;
  }
}

registerProcessor("audio-processor", AudioProcessor);
