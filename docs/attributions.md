## 🙏 Благодарности / Acknowledgements

- **Qur'an text** — King Fahd Glorious Qur'an Printing Complex (KFGQPC) scripts (`qpc_hafs`).
- **Reciter timecodes** (`data/references/`) — community datasets
  ([quran.com](https://quran.com), [Tarteel QUL](https://qul.tarteel.ai/),
  [quran-align](https://github.com/cpfair/quran-align)), typically CC-BY 4.0.
- **Forced alignment** — [ctc-forced-aligner](https://github.com/MahmoudAshraf97/ctc-forced-aligner),
  [ctc-segmentation](https://github.com/lumaku/ctc-segmentation), and prior art from
  [lafzize](https://sr.ht/~rehandaphedar/lafzize/).
- **Acoustic models** — Hugging Face community (e.g. `TBOGamer22/wav2vec2-quran-phonetics`,
  MMS forced aligner), Silero VAD.
- **Tarteel Whisper Quran** — [Tiny](https://huggingface.co/tarteel-ai/whisper-tiny-ar-quran)
  and [Base](https://huggingface.co/tarteel-ai/whisper-base-ar-quran), Apache-2.0.
  Explicit generation/alignment metadata from the original OpenAI Tiny/Base,
  with source revisions and SHA256 recorded in `core/model_catalog.json`.
- **Font** — KFGQPC HAFS Uthmanic Script.
- **Surah navigation names** — [Tanzil Quran Metadata](https://tanzil.net/docs/Quran_Metadata),
  including English transliterations. These labels do not supply Quran text or ayah counts.

Pull in upstream attributions when redistributing. See [LICENSE](../LICENSE).

## 📜 Лицензия / License

[MIT](../LICENSE) для кода. Вендоренные данные, шрифт и модели сохраняют лицензии источников —
см. примечание в [LICENSE](../LICENSE).
