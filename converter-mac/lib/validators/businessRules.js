function validateBusinessRules(invoice) {
  const errors = [];

  // Mandatory fields
  if (!invoice.id) {
    errors.push({ field: 'id', message: 'Invoice number is required (BT-1)', severity: 'CRITICAL' });
  }
  if (!invoice.issueDate) {
    errors.push({ field: 'issueDate', message: 'Issue date is required (BT-2)', severity: 'CRITICAL' });
  }
  if (!invoice.seller?.name) {
    errors.push({ field: 'seller.name', message: 'Seller name is required (BT-27)', severity: 'CRITICAL' });
  }
  if (!invoice.seller?.address?.country) {
    errors.push({ field: 'seller.address.country', message: 'Seller country is required (BT-40)', severity: 'CRITICAL' });
  }
  if (!invoice.seller?.vatId) {
    errors.push({ field: 'seller.vatId', message: 'Seller VAT ID is required (BT-31)', severity: 'CRITICAL' });
  }
  if (!invoice.buyer?.name) {
    errors.push({ field: 'buyer.name', message: 'Buyer name is required (BT-44)', severity: 'CRITICAL' });
  }
  if (!invoice.buyer?.address?.country) {
    errors.push({ field: 'buyer.address.country', message: 'Buyer country is required (BT-55)', severity: 'CRITICAL' });
  }
  if (!invoice.lines || invoice.lines.length === 0) {
    errors.push({ field: 'lines', message: 'At least one invoice line is required (BG-25)', severity: 'CRITICAL' });
  }
  if (!invoice.payment?.currency) {
    errors.push({ field: 'payment.currency', message: 'Currency is required (BT-5)', severity: 'CRITICAL' });
  }

  // Line item validation
  if (invoice.lines) {
    invoice.lines.forEach((line, idx) => {
      if (!line.description) {
        errors.push({ field: `lines[${idx}].description`, message: `Line ${idx + 1}: Description required`, severity: 'CRITICAL' });
      }
      if (line.quantity === undefined || line.quantity === null) {
        errors.push({ field: `lines[${idx}].quantity`, message: `Line ${idx + 1}: Quantity required`, severity: 'CRITICAL' });
      }
      if (line.unitPrice === undefined || line.unitPrice === null) {
        errors.push({ field: `lines[${idx}].unitPrice`, message: `Line ${idx + 1}: Unit price required`, severity: 'CRITICAL' });
      }
    });
  }

  // Calculation validation
  if (invoice.totals) {
    const calculatedGross = invoice.totals.netTotal + invoice.totals.taxTotal;
    if (Math.abs(calculatedGross - invoice.totals.grossTotal) > 0.01) {
      errors.push({
        field: 'totals.grossTotal',
        message: `Gross total mismatch: expected ${calculatedGross.toFixed(2)}, got ${invoice.totals.grossTotal.toFixed(2)}`,
        severity: 'CRITICAL',
        suggestion: 'Ensure grossTotal = netTotal + taxTotal',
      });
    }
  }

  // Tax validation
  if (invoice.taxes) {
    invoice.taxes.forEach((tax, idx) => {
      if (tax.rate < 0 || tax.rate > 100) {
        errors.push({
          field: `taxes[${idx}].rate`,
          message: `Invalid tax rate: ${tax.rate}%`,
          severity: 'WARNING',
        });
      }
    });
  }

  // Date format validation
  const dateRegex = /^\d{4}-\d{2}-\d{2}$/;
  if (invoice.issueDate && !dateRegex.test(invoice.issueDate)) {
    errors.push({ field: 'issueDate', message: 'Issue date must be YYYY-MM-DD', severity: 'WARNING' });
  }

  // Country code validation
  const countryCodeRegex = /^[A-Z]{2}$/;
  if (invoice.seller?.address?.country && !countryCodeRegex.test(invoice.seller.address.country)) {
    errors.push({ field: 'seller.address.country', message: 'Country must be 2-letter ISO code', severity: 'WARNING' });
  }

  // Currency code validation
  const currencyCodeRegex = /^[A-Z]{3}$/;
  if (invoice.payment?.currency && !currencyCodeRegex.test(invoice.payment.currency)) {
    errors.push({ field: 'payment.currency', message: 'Currency must be 3-letter ISO code', severity: 'WARNING' });
  }

  return errors;
}

module.exports = { validateBusinessRules };
