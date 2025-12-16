// EN16931 Invoice Model - Type definitions for reference
// This file documents the expected structure

/*
Invoice Structure:
{
  id: string,                    // BT-1: Invoice number
  issueDate: string,             // BT-2: YYYY-MM-DD
  dueDate?: string,              // BT-9
  invoiceType?: string,          // BT-3: Default "380"
  
  seller: {
    name: string,                // BT-27
    address: {
      street: string,            // BT-35
      city: string,              // BT-37
      postalCode: string,        // BT-38
      country: string,           // BT-40: ISO 3166-1
    },
    vatId: string,               // BT-31
    taxScheme: "VAT",
  },
  
  buyer: {
    name: string,                // BT-44
    address: {
      street: string,
      city: string,
      postalCode: string,
      country: string,
    },
    vatId?: string,
  },
  
  lines: [{
    id: string,
    description: string,
    quantity: number,
    unit: string,
    unitPrice: number,
    lineTotal: number,
    tax: {
      category: string,          // S, Z, E
      rate: number,
      amount: number,
    }
  }],
  
  totals: {
    lineTotal: number,
    allowanceTotal: number,
    chargeTotal: number,
    taxTotal: number,
    netTotal: number,
    grossTotal: number,
    amountDue: number,
  },
  
  taxes: [{
    category: string,
    rate: number,
    taxableAmount: number,
    taxAmount: number,
  }],
  
  payment: {
    currency: string,            // BT-5: ISO 4217
    iban?: string,
    bic?: string,
  }
}
*/

module.exports = {};
