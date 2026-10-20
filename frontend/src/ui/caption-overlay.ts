import { DownstreamEventPacket } from "../contracts/types";

export class CaptionOverlay {
  private readonly el: HTMLDivElement;

  constructor(root: HTMLElement) {
    this.el = document.createElement("div");
    root.appendChild(this.el);
  }

  public render(pkt: DownstreamEventPacket): void {
    this.el.replaceChildren();

    const text = pkt.per_speaker_target_texts.join("\n");
    const displayText = pkt.vocal_modes.includes(2) ? `${text} ♪` : text;

    if (pkt.codeswitch_words.length === 0) {
      this.el.appendChild(document.createTextNode(displayText));
    } else {
      const escapedWords = pkt.codeswitch_words.map((w) =>
        w.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")
      );
      const regex = new RegExp(`(${escapedWords.join("|")})`, "gi");
      const parts = displayText.split(regex);

      for (const part of parts) {
        if (
          pkt.codeswitch_words.some(
            (w) => w.toLowerCase() === part.toLowerCase()
          )
        ) {
          const mark = document.createElement("mark");
          mark.textContent = part;
          this.el.appendChild(mark);
        } else if (part.length > 0) {
          this.el.appendChild(document.createTextNode(part));
        }
      }
    }
    this.el.dataset.driftMs = String(pkt.drift_ms);
  }
}
