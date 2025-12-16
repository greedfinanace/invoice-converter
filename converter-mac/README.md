# Invoice Converter for macOS

A professional desktop application for converting invoices between EN16931 compliant formats.

---

## Overview

Invoice Converter is a lightweight, user-friendly tool designed to transform invoice files between industry-standard formats. Built for compliance with the European EN16931 e-invoicing standard, it supports seamless conversion between UBL 2.1, UN/CEFACT CII, PDF, and JSON formats.

---

## Features

- Native macOS experience with dark mode support
- Drag and drop file support
- Real-time EN16931 business rule validation
- Multiple input and output format support
- Country-specific CIUS (Core Invoice Usage Specification) compliance
- Automatic output file naming and placement
- No internet connection required
- Universal binary (Apple Silicon & Intel)

---

## Installation

### DMG Installation

1. Download `Invoice Converter.dmg`
2. Open the DMG file
3. Drag Invoice Converter to your Applications folder
4. Launch from Applications or Spotlight

### First Launch

On first launch, macOS may show a security warning. To open:
1. Right-click (or Control-click) the app
2. Select "Open" from the context menu
3. Click "Open" in the dialog

---

## Usage

1. Launch Invoice Converter
2. Drag and drop your invoice file onto the application window
   - Or use File > Open Invoice... (⌘O)
3. Select the desired output format
4. Choose the appropriate country for CIUS rules
5. Click "Convert"

The converted file will be saved in the same directory as the source file with the naming convention:

    [original_filename]_converted_[FORMAT].[extension]

---

## Supported Formats

### Input Formats

| Format | Description                              |
|--------|------------------------------------------|
| JSON   | EN16931 structured invoice data          |
| XML    | UBL 2.1 or CII (automatically detected)  |
| CSV    | Spreadsheet format with metadata header  |

### Output Formats

| Format  | Description                                    |
|---------|------------------------------------------------|
| UBL 2.1 | OASIS Universal Business Language XML          |
| CII     | UN/CEFACT Cross-Industry Invoice XML           |
| PDF     | Human-readable document (HTML-based)           |
| JSON    | Structured data format                         |

---

## Country Support

The application supports country-specific CIUS implementations:

| Country        | Standard      |
|----------------|---------------|
| Germany        | XRechnung     |
| France         | Factur-X      |
| Italy          | FatturaPA     |
| Spain          | -             |
| United Kingdom | -             |
| Austria        | -             |
| Netherlands    | -             |
| Belgium        | -             |

---

## System Requirements

- macOS 10.13 (High Sierra) or later
- Apple Silicon (M1/M2/M3) or Intel processor
- 100 MB disk space

---

## License

This software is provided under a custom license with the following terms:

**Permitted Use:**
- Personal, non-commercial use
- Business use by organizations with capital under $15 million USD

**Restricted Use:**
- Organizations with capital exceeding $15 million USD require written permission
- Modification of the software is prohibited
- Redistribution without permission is prohibited

See `LICENSE.txt` for complete terms and conditions.

---

## Contact

For licensing inquiries, enterprise permissions, or support:

**Email:** greedthefirst@gmail.com

---

## Technical Information

- Platform: macOS (Universal Binary)
- Framework: Electron
- Standards: EN16931, UBL 2.1, UN/CEFACT CII

---

Invoice Converter - Professional EN16931 Format Conversion
