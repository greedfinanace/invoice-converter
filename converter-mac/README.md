# Invoice Converter for macOS

<p align="center">
  <img src="assets/icon.png" alt="Invoice Converter" width="128" height="128">
</p>

<p align="center">
  <strong>Professional EN16931 Invoice Format Conversion</strong><br>
  Built natively for Mac • Apple Silicon & Intel supported
</p>

---

## What is this?

Invoice Converter transforms your invoice files between industry-standard e-invoicing formats. Whether you're dealing with UBL, CII, JSON, or need a printable PDF — this app handles it all while staying compliant with the European EN16931 standard.

---

## Features

- 🍎 **Native macOS Experience** — Feels right at home on your Mac
- 🌙 **Dark Mode** — Easy on the eyes, day or night
- 📁 **Drag & Drop** — Just drop your file and go
- ✅ **Real-time Validation** — Catch errors before they become problems
- 🌍 **Country-specific Rules** — XRechnung, Factur-X, FatturaPA support
- 🔒 **Offline** — Your data never leaves your Mac
- ⚡ **Universal Binary** — Runs natively on M1/M2/M3 and Intel Macs

---

## Installation

### Option 1: DMG Installer (Recommended)

1. Download `Invoice Converter.dmg`
2. Double-click to open
3. Drag **Invoice Converter** to your **Applications** folder
4. Launch from Applications or Spotlight (⌘ + Space)

### Option 2: ZIP Archive

1. Download `Invoice Converter-mac.zip`
2. Extract the archive
3. Move `Invoice Converter.app` to Applications
4. Launch and enjoy

### First Launch Security

macOS may show a security prompt on first launch:

1. **Right-click** (or Control-click) the app
2. Select **"Open"** from the menu
3. Click **"Open"** in the dialog

This is only needed once.

---

## How to Use

| Step | Action |
|------|--------|
| 1 | Launch Invoice Converter |
| 2 | Drag your invoice file onto the window (or use ⌘O) |
| 3 | Pick your output format |
| 4 | Select your country for CIUS rules |
| 5 | Click **Convert** |

Your converted file appears in the same folder as the original:
```
MyInvoice.json → MyInvoice_converted_UBL.xml
```

---

## Supported Formats

### Input
| Format | Description |
|--------|-------------|
| **JSON** | EN16931 structured data |
| **XML** | UBL 2.1 or CII (auto-detected) |
| **CSV** | Spreadsheet with metadata header |

### Output
| Format | Description |
|--------|-------------|
| **UBL 2.1** | OASIS Universal Business Language |
| **CII** | UN/CEFACT Cross-Industry Invoice |
| **PDF** | Human-readable HTML document |
| **JSON** | Structured data format |

---

## Country Support

| Country | Standard | Status |
|---------|----------|--------|
| 🇩🇪 Germany | XRechnung | ✅ Full |
| 🇫🇷 France | Factur-X | ✅ Full |
| 🇮🇹 Italy | FatturaPA | ✅ Full |
| 🇪🇸 Spain | — | ✅ Supported |
| 🇬🇧 United Kingdom | — | ✅ Supported |
| 🇦🇹 Austria | — | ✅ Supported |
| 🇳🇱 Netherlands | — | ✅ Supported |
| 🇧🇪 Belgium | — | ✅ Supported |

---

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| ⌘ O | Open invoice file |
| ⌘ W | Close window |
| ⌘ Q | Quit app |
| ⌘ , | Preferences |
| ⌘ + | Zoom in |
| ⌘ - | Zoom out |

---

## System Requirements

- **macOS** 10.13 High Sierra or later
- **Processor** Apple Silicon (M1/M2/M3) or Intel
- **Disk Space** ~150 MB

---

## Building from Source

```bash
# Clone or download the source
cd converter-mac

# Install dependencies
npm install

# Run in development
npm start

# Build for distribution
npm run build:mac
```

Build outputs appear in the `dist/` folder.

---

## License

**Free for:**
- Personal use
- Businesses with capital under $15 million USD

**Requires permission:**
- Organizations with capital over $15 million USD
- Redistribution or modification

See [LICENSE.txt](LICENSE.txt) for full terms.

---

## Support & Contact

📧 **Email:** greedthefirst@gmail.com

For licensing inquiries, enterprise permissions, or support questions.

---

<p align="center">
  <sub>Invoice Converter • EN16931 Compliant • Made for Mac</sub>
</p>
