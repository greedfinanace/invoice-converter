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

- Native macOS experience — feels right at home on your Mac
- Dark mode support — easy on the eyes, day or night
- Drag and drop — just drop your file and go
- Real-time validation — catch errors before they become problems
- Country-specific rules — XRechnung, Factur-X, FatturaPA support
- Fully offline — your data never leaves your Mac
- Universal binary — runs natively on M1/M2/M3 and Intel Macs

---

## Getting Started (Complete Beginner Guide)

### Step 1: Install Node.js (if you don't have it)

Node.js is required to build the app. Here's how to get it:

1. Go to **nodejs.org**
2. Click the big green **"LTS"** button (recommended version)
3. Open the downloaded `.pkg` file
4. Follow the installer — just keep clicking **Continue** and **Install**
5. Done! Node.js is now installed

**To verify it worked:**
1. Open **Terminal** (press Cmd + Space, type "Terminal", hit Enter)
2. Type `node --version` and press Enter
3. You should see something like `v20.10.0` — that means it's working!

---

### Step 2: Open Terminal in this folder

1. Open **Finder**
2. Navigate to the `converter-mac` folder
3. **Right-click** on the folder
4. Select **"New Terminal at Folder"**

**Don't see that option?** Do this instead:
1. Open **Terminal** (press Cmd + Space, type "Terminal", hit Enter)
2. Type `cd ` (with a space after it)
3. Drag the `converter-mac` folder into Terminal
4. Press **Enter**

---

### Step 3: Install the app's dependencies

In Terminal, type this and press Enter:

```bash
npm install
```

Wait for it to finish (might take 1-2 minutes). You'll see a lot of text scrolling — that's normal!

---

### Step 4: Build the app

Now type this and press Enter:

```bash
npm run build:mac
```

This takes a few minutes. When it's done, you'll have your app!

---

### Step 5: Find your app

1. Open **Finder**
2. Go to the `converter-mac` folder
3. Open the `dist` folder inside it
4. You'll see:
   - `Invoice Converter.dmg` — This is your installer!
   - `mac` folder — Contains the `.app` file

---

## Installing the App

### From the DMG (Recommended)

1. **Double-click** `Invoice Converter.dmg`
2. A window opens with the app and an Applications folder
3. **Drag** the Invoice Converter icon **onto** the Applications folder
4. Wait for it to copy
5. **Eject** the DMG (click the eject icon next to it in Finder sidebar)
6. Open **Applications** folder and double-click **Invoice Converter**

### First Time Opening

macOS might say the app is from an "unidentified developer". Here's how to open it:

1. **Don't** click "Move to Trash"!
2. Open **System Preferences** then **Security & Privacy**
3. Click **"Open Anyway"** next to the Invoice Converter message
4. Or: **Right-click** the app, then **Open**, then **Open**

You only need to do this once.

---

## How to Use the App

### Converting an Invoice

1. **Open** Invoice Converter
2. **Drag** your invoice file onto the app window
   - Or click the drop zone to browse for a file
   - Or press Cmd + O to open a file
3. **Choose** your output format:
   - **UBL 2.1** — Standard XML format
   - **CII** — UN/CEFACT format
   - **PDF** — Printable document
   - **JSON** — Data format
4. **Select** your country (for country-specific rules)
5. Click **Convert**
6. Done! Your converted file is saved next to the original

### Where's my converted file?

It's in the **same folder** as your original file, with a new name:
```
Original:   MyInvoice.json
Converted:  MyInvoice_converted_UBL.xml
```

---

## What Files Can I Convert?

### Files You Can Open (Input)

| Type | What it is |
|------|------------|
| .json | Invoice data in JSON format |
| .xml | UBL or CII invoice (auto-detected) |
| .csv | Spreadsheet/Excel export |

### Files You Can Create (Output)

| Type | What it is |
|------|------------|
| UBL 2.1 | European standard XML |
| CII | UN/CEFACT standard XML |
| PDF | Printable invoice document |
| JSON | Structured data file |

---

## Supported Countries

| Country | Standard |
|---------|----------|
| Germany | XRechnung |
| France | Factur-X |
| Italy | FatturaPA |
| Spain | Standard |
| United Kingdom | Standard |
| Austria | Standard |
| Netherlands | Standard |
| Belgium | Standard |

---

## Keyboard Shortcuts

| Press | To do this |
|-------|------------|
| Cmd + O | Open a file |
| Cmd + W | Close window |
| Cmd + Q | Quit the app |
| Cmd + Plus | Zoom in |
| Cmd + Minus | Zoom out |

---

## System Requirements

- **Mac:** macOS 10.13 (High Sierra) or newer
- **Chip:** Apple Silicon (M1/M2/M3) or Intel — both work!
- **Space:** About 150 MB

---

## Troubleshooting

### "npm: command not found"
Node.js isn't installed. Go back to Step 1.

### "Cannot be opened because it is from an unidentified developer"
See the "First Time Opening" section above.

### The build failed
Make sure you're in the right folder. In Terminal, type:
```bash
pwd
```
It should end with `converter-mac`. If not, go back to Step 2.

### Something else went wrong
Try deleting the node_modules folder and starting over:
```bash
rm -rf node_modules
npm install
npm run build:mac
```

---

## License

**Free for:**
- Personal use
- Businesses with capital under $15 million USD

**Need permission for:**
- Large organizations (capital over $15 million USD)
- Redistribution or modification

See [LICENSE.txt](LICENSE.txt) for full terms.

---

## Need Help?

**Email:** greedthefirst@gmail.com

For licensing questions, enterprise permissions, or support.

---

<p align="center">
  <sub>Invoice Converter • EN16931 Compliant • Made for Mac</sub>
</p>
