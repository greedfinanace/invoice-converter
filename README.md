# Invoice Converter

A professional desktop application for converting invoices between EN16931 compliant formats.

---

## Overview

Invoice Converter is a lightweight, user-friendly tool designed to transform invoice files between industry-standard formats. Built for compliance with the European EN16931 e-invoicing standard, it supports seamless conversion between UBL 2.1, UN/CEFACT CII, PDF, and JSON formats.

---

## Features

- Clean, modern dark-themed interface
- Drag and drop file support
- Real-time EN16931 business rule validation
- Multiple input and output format support
- Country-specific CIUS (Core Invoice Usage Specification) compliance
- Automatic output file naming and placement
- No internet connection required

---

## Installation

### Standard Installation

Run `Invoice Converter Setup.exe` to launch the installation wizard:

1. Accept the license agreement
2. Choose your installation directory (default: Program Files)
3. Complete the installation

The installer will create:
- Desktop shortcut
- Start Menu entry
- Uninstaller (accessible via Add/Remove Programs)

### Portable Version

Alternatively, use `Invoice Converter.exe` directly without installation.

---

## Usage

1. Launch Invoice Converter
2. Drag and drop your invoice file onto the application window
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

- Platform: Windows (64-bit)
- Framework: Electron
- Standards: EN16931, UBL 2.1, UN/CEFACT CII

---

Invoice Converter - Professional EN16931 Format Conversion
