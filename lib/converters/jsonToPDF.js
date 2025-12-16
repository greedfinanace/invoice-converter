const Decimal = require('decimal.js');

function jsonToPDF(invoice) {
  const currency = invoice.payment?.currency || 'EUR';

  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Invoice ${invoice.id}</title>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; color: #333; line-height: 1.6; padding: 40px; background: #fff; }
    .container { max-width: 800px; margin: 0 auto; }
    .header { display: flex; justify-content: space-between; margin-bottom: 40px; border-bottom: 3px solid #1e40af; padding-bottom: 20px; }
    .header-left h1 { font-size: 32px; color: #1e40af; margin-bottom: 5px; }
    .header-right { text-align: right; }
    .header-right p { font-size: 24px; font-weight: 600; color: #1e40af; }
    .invoice-details { display: flex; justify-content: space-between; margin-bottom: 30px; }
    .invoice-details label { font-weight: 600; color: #666; font-size: 12px; text-transform: uppercase; }
    .invoice-details p { margin-top: 5px; font-size: 14px; }
    .parties { display: flex; justify-content: space-between; margin-bottom: 40px; }
    .party { width: 45%; }
    .party h3 { font-size: 12px; font-weight: 600; color: #666; text-transform: uppercase; margin-bottom: 10px; border-bottom: 1px solid #ddd; padding-bottom: 5px; }
    .party p { margin: 3px 0; font-size: 14px; }
    .party .name { font-weight: 600; font-size: 16px; }
    table { width: 100%; border-collapse: collapse; margin-bottom: 30px; }
    th { background-color: #1e40af; color: white; font-weight: 600; padding: 12px 10px; text-align: left; font-size: 12px; text-transform: uppercase; }
    td { padding: 12px 10px; border-bottom: 1px solid #e5e7eb; font-size: 14px; }
    tr:nth-child(even) { background-color: #f9fafb; }
    .text-right { text-align: right; }
    .totals { width: 300px; margin-left: auto; }
    .totals-row { display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #e5e7eb; }
    .totals-row.grand { border-top: 2px solid #1e40af; border-bottom: 2px solid #1e40af; margin-top: 10px; padding: 12px 0; }
    .totals-row.grand .label, .totals-row.grand .amount { font-size: 18px; font-weight: 700; color: #1e40af; }
    .payment-info { margin-top: 30px; padding: 20px; background-color: #f3f4f6; border-radius: 8px; }
    .payment-info h3 { font-size: 14px; font-weight: 600; margin-bottom: 10px; }
    .payment-info p { font-size: 13px; margin: 5px 0; }
    .footer { margin-top: 40px; padding-top: 20px; border-top: 1px solid #e5e7eb; font-size: 11px; color: #999; text-align: center; }
    @media print { body { padding: 20px; } }
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <div class="header-left">
        <h1>INVOICE</h1>
        <p style="color: #666; font-size: 14px;">EN16931 Compliant</p>
      </div>
      <div class="header-right">
        <p>${invoice.id}</p>
      </div>
    </div>

    <div class="invoice-details">
      <div>
        <label>Issue Date</label>
        <p>${invoice.issueDate}</p>
      </div>
      <div>
        <label>Due Date</label>
        <p>${invoice.dueDate || 'N/A'}</p>
      </div>
      <div>
        <label>Currency</label>
        <p>${currency}</p>
      </div>
    </div>

    <div class="parties">
      <div class="party">
        <h3>From (Seller)</h3>
        <p class="name">${invoice.seller.name}</p>
        <p>${invoice.seller.address.street}</p>
        <p>${invoice.seller.address.postalCode} ${invoice.seller.address.city}</p>
        <p>${invoice.seller.address.country}</p>
        <p style="margin-top: 10px;"><strong>VAT ID:</strong> ${invoice.seller.vatId}</p>
      </div>
      <div class="party">
        <h3>To (Buyer)</h3>
        <p class="name">${invoice.buyer.name}</p>
        <p>${invoice.buyer.address.street}</p>
        <p>${invoice.buyer.address.postalCode} ${invoice.buyer.address.city}</p>
        <p>${invoice.buyer.address.country}</p>
        ${invoice.buyer.vatId ? `<p style="margin-top: 10px;"><strong>VAT ID:</strong> ${invoice.buyer.vatId}</p>` : ''}
      </div>
    </div>

    <table>
      <thead>
        <tr>
          <th style="width: 40%;">Description</th>
          <th class="text-right">Qty</th>
          <th class="text-right">Unit Price</th>
          <th class="text-right">Tax</th>
          <th class="text-right">Amount</th>
        </tr>
      </thead>
      <tbody>
        ${invoice.lines.map(line => `
          <tr>
            <td>${line.description}</td>
            <td class="text-right">${line.quantity} ${line.unit}</td>
            <td class="text-right">${new Decimal(line.unitPrice).toFixed(2)} ${currency}</td>
            <td class="text-right">${line.tax.rate}%</td>
            <td class="text-right">${new Decimal(line.lineTotal).toFixed(2)} ${currency}</td>
          </tr>
        `).join('')}
      </tbody>
    </table>

    <div class="totals">
      <div class="totals-row">
        <span class="label">Subtotal:</span>
        <span class="amount">${new Decimal(invoice.totals.netTotal).toFixed(2)} ${currency}</span>
      </div>
      <div class="totals-row">
        <span class="label">Tax (VAT):</span>
        <span class="amount">${new Decimal(invoice.totals.taxTotal).toFixed(2)} ${currency}</span>
      </div>
      <div class="totals-row grand">
        <span class="label">Total Due:</span>
        <span class="amount">${new Decimal(invoice.totals.grossTotal).toFixed(2)} ${currency}</span>
      </div>
    </div>

    ${invoice.payment?.iban ? `
      <div class="payment-info">
        <h3>Payment Details</h3>
        <p><strong>IBAN:</strong> ${invoice.payment.iban}</p>
        ${invoice.payment.bic ? `<p><strong>BIC:</strong> ${invoice.payment.bic}</p>` : ''}
      </div>
    ` : ''}

    <div class="footer">
      <p>Generated on ${new Date().toLocaleString()} | EN16931 Compliant Invoice</p>
    </div>
  </div>
</body>
</html>`;
}

module.exports = { jsonToPDF };
