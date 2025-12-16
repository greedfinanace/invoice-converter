const Papa = require('papaparse');

async function csvToJSON(csvContent) {
  return new Promise((resolve, reject) => {
    Papa.parse(csvContent, {
      header: true,
      skipEmptyLines: true,
      complete: (results) => {
        try {
          const data = results.data;
          
          if (data.length === 0) {
            throw new Error('CSV is empty');
          }

          // First row contains seller/buyer info (metadata)
          const metadata = data[0];

          // Rest are line items
          const lineItems = data.slice(1);

          const lines = lineItems
            .filter(row => row.Description && row.Description.trim())
            .map((row, idx) => ({
              id: String(idx + 1),
              description: row.Description || '',
              quantity: parseFloat(row.Quantity || '0'),
              unit: row.Unit || 'EA',
              unitPrice: parseFloat(row['Unit Price'] || '0'),
              lineTotal: parseFloat(row['Line Total'] || 
                (parseFloat(row.Quantity || '0') * parseFloat(row['Unit Price'] || '0')).toString()),
              tax: {
                category: row['Tax Category'] || 'S',
                rate: parseFloat(row['Tax Rate'] || '19'),
                amount: 0,
              },
            }));

          // Calculate tax amounts
          const currency = metadata.Currency || 'EUR';
          for (const line of lines) {
            line.tax.amount = (line.lineTotal * line.tax.rate) / 100;
          }

          // Build invoice
          const invoice = {
            id: metadata['Invoice Number'] || `INV-${Date.now()}`,
            issueDate: metadata['Issue Date'] || new Date().toISOString().split('T')[0],
            dueDate: metadata['Due Date'] || '',
            invoiceType: '380',
            seller: {
              name: metadata['Seller Name'] || '',
              address: {
                street: metadata['Seller Street'] || '',
                city: metadata['Seller City'] || '',
                postalCode: metadata['Seller Postal Code'] || '',
                country: metadata['Seller Country'] || 'DE',
              },
              vatId: metadata['Seller VAT ID'] || '',
              taxScheme: 'VAT',
            },
            buyer: {
              name: metadata['Buyer Name'] || '',
              address: {
                street: metadata['Buyer Street'] || '',
                city: metadata['Buyer City'] || '',
                postalCode: metadata['Buyer Postal Code'] || '',
                country: metadata['Buyer Country'] || 'DE',
              },
              vatId: metadata['Buyer VAT ID'],
              taxScheme: 'VAT',
            },
            lines,
            totals: {
              lineTotal: lines.reduce((sum, l) => sum + l.lineTotal, 0),
              allowanceTotal: 0,
              chargeTotal: 0,
              taxTotal: lines.reduce((sum, l) => sum + l.tax.amount, 0),
              netTotal: lines.reduce((sum, l) => sum + l.lineTotal, 0),
              grossTotal: lines.reduce((sum, l) => sum + l.lineTotal + l.tax.amount, 0),
              amountDue: lines.reduce((sum, l) => sum + l.lineTotal + l.tax.amount, 0),
            },
            taxes: Array.from(new Set(lines.map(l => l.tax.category))).map(cat => {
              const catLines = lines.filter(l => l.tax.category === cat);
              return {
                category: cat,
                rate: catLines[0].tax.rate,
                taxableAmount: catLines.reduce((sum, l) => sum + l.lineTotal, 0),
                taxAmount: catLines.reduce((sum, l) => sum + l.tax.amount, 0),
              };
            }),
            payment: {
              currency,
              iban: metadata.IBAN,
              bic: metadata.BIC,
            },
          };

          resolve(invoice);
        } catch (error) {
          reject(error);
        }
      },
      error: (error) => reject(error),
    });
  });
}

module.exports = { csvToJSON };
