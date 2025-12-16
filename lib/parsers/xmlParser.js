const { parseStringPromise } = require('xml2js');

async function xmlToJSON(xmlString, format) {
  const parsed = await parseStringPromise(xmlString, {
    explicitArray: false,
    ignoreAttrs: false,
    tagNameProcessors: [(name) => name.replace(/^[a-z]+:/i, '')],
  });

  if (format === 'ubl') {
    return parseUBL(parsed);
  } else {
    return parseCII(parsed);
  }
}

function getText(obj) {
  if (!obj) return '';
  if (typeof obj === 'string') return obj;
  if (obj._) return obj._;
  if (obj['#text']) return obj['#text'];
  return '';
}

function getNumber(obj) {
  const text = getText(obj);
  return parseFloat(text) || 0;
}

function parseUBL(parsed) {
  const inv = parsed.Invoice || parsed;

  const lines = [];
  const invoiceLines = Array.isArray(inv.InvoiceLine) ? inv.InvoiceLine : [inv.InvoiceLine].filter(Boolean);

  for (const line of invoiceLines) {
    const item = line.Item || {};
    const price = line.Price || {};
    const taxCategory = item.ClassifiedTaxCategory || {};

    lines.push({
      id: getText(line.ID),
      description: getText(item.Name),
      quantity: getNumber(line.InvoicedQuantity),
      unit: line.InvoicedQuantity?.$?.unitCode || 'EA',
      unitPrice: getNumber(price.PriceAmount),
      lineTotal: getNumber(line.LineExtensionAmount),
      tax: {
        category: getText(taxCategory.ID) || 'S',
        rate: getNumber(taxCategory.Percent),
        amount: 0,
      },
    });
  }

  // Calculate tax amounts
  for (const line of lines) {
    line.tax.amount = (line.lineTotal * line.tax.rate) / 100;
  }

  const supplierParty = inv.AccountingSupplierParty?.Party || {};
  const customerParty = inv.AccountingCustomerParty?.Party || {};
  const monetary = inv.LegalMonetaryTotal || {};
  const taxTotal = inv.TaxTotal || {};

  const taxes = [];
  const taxSubtotals = Array.isArray(taxTotal.TaxSubtotal) ? taxTotal.TaxSubtotal : [taxTotal.TaxSubtotal].filter(Boolean);
  for (const sub of taxSubtotals) {
    const cat = sub.TaxCategory || {};
    taxes.push({
      category: getText(cat.ID) || 'S',
      rate: getNumber(cat.Percent),
      taxableAmount: getNumber(sub.TaxableAmount),
      taxAmount: getNumber(sub.TaxAmount),
    });
  }


  return {
    id: getText(inv.ID),
    issueDate: getText(inv.IssueDate),
    dueDate: getText(inv.DueDate),
    invoiceType: getText(inv.InvoiceTypeCode) || '380',
    seller: {
      name: getText(supplierParty.PartyName?.Name),
      address: {
        street: getText(supplierParty.PostalAddress?.StreetName),
        city: getText(supplierParty.PostalAddress?.CityName),
        postalCode: getText(supplierParty.PostalAddress?.PostalZone),
        country: getText(supplierParty.PostalAddress?.Country?.IdentificationCode),
      },
      vatId: getText(supplierParty.PartyTaxScheme?.CompanyID),
      taxScheme: 'VAT',
    },
    buyer: {
      name: getText(customerParty.PartyName?.Name),
      address: {
        street: getText(customerParty.PostalAddress?.StreetName),
        city: getText(customerParty.PostalAddress?.CityName),
        postalCode: getText(customerParty.PostalAddress?.PostalZone),
        country: getText(customerParty.PostalAddress?.Country?.IdentificationCode),
      },
      vatId: getText(customerParty.PartyTaxScheme?.CompanyID),
      taxScheme: 'VAT',
    },
    lines,
    totals: {
      lineTotal: getNumber(monetary.LineExtensionAmount),
      allowanceTotal: getNumber(monetary.AllowanceTotalAmount),
      chargeTotal: getNumber(monetary.ChargeTotalAmount),
      taxTotal: getNumber(taxTotal.TaxAmount),
      netTotal: getNumber(monetary.TaxExclusiveAmount),
      grossTotal: getNumber(monetary.TaxInclusiveAmount),
      amountDue: getNumber(monetary.PayableAmount),
    },
    taxes,
    payment: {
      currency: getText(inv.DocumentCurrencyCode) || 'EUR',
    },
  };
}

function parseCII(parsed) {
  const inv = parsed.CrossIndustryInvoice || parsed;
  const doc = inv.ExchangedDocument || {};
  const transaction = inv.SupplyChainTradeTransaction || {};
  const agreement = transaction.ApplicableHeaderTradeAgreement || {};
  const settlement = transaction.ApplicableHeaderTradeSettlement || {};
  const monetary = settlement.SpecifiedTradeSettlementHeaderMonetarySummation || {};

  const seller = agreement.SellerTradeParty || {};
  const buyer = agreement.BuyerTradeParty || {};

  const lines = [];
  const lineItems = Array.isArray(transaction.IncludedSupplyChainTradeLineItem)
    ? transaction.IncludedSupplyChainTradeLineItem
    : [transaction.IncludedSupplyChainTradeLineItem].filter(Boolean);

  for (const item of lineItems) {
    const lineDoc = item.AssociatedDocumentLineDocument || {};
    const product = item.SpecifiedTradeProduct || {};
    const lineAgreement = item.SpecifiedLineTradeAgreement || {};
    const lineDelivery = item.SpecifiedLineTradeDelivery || {};
    const lineSettlement = item.SpecifiedLineTradeSettlement || {};
    const lineTax = lineSettlement.ApplicableTradeTax || {};
    const lineMonetary = lineSettlement.SpecifiedTradeSettlementLineMonetarySummation || {};

    lines.push({
      id: getText(lineDoc.LineID),
      description: getText(product.Name),
      quantity: getNumber(lineDelivery.BilledQuantity),
      unit: lineDelivery.BilledQuantity?.$?.unitCode || 'EA',
      unitPrice: getNumber(lineAgreement.NetPriceProductTradePrice?.ChargeAmount),
      lineTotal: getNumber(lineMonetary.LineTotalAmount),
      tax: {
        category: getText(lineTax.CategoryCode) || 'S',
        rate: getNumber(lineTax.RateApplicablePercent),
        amount: 0,
      },
    });
  }

  for (const line of lines) {
    line.tax.amount = (line.lineTotal * line.tax.rate) / 100;
  }

  const taxes = [];
  const tradeTaxes = Array.isArray(settlement.ApplicableTradeTax)
    ? settlement.ApplicableTradeTax
    : [settlement.ApplicableTradeTax].filter(Boolean);

  for (const tax of tradeTaxes) {
    taxes.push({
      category: getText(tax.CategoryCode) || 'S',
      rate: getNumber(tax.RateApplicablePercent),
      taxableAmount: getNumber(tax.BasisAmount),
      taxAmount: getNumber(tax.CalculatedAmount),
    });
  }

  const issueDateStr = getText(doc.IssueDateTime?.DateTimeString);
  const issueDate = issueDateStr.length === 8
    ? `${issueDateStr.slice(0, 4)}-${issueDateStr.slice(4, 6)}-${issueDateStr.slice(6, 8)}`
    : issueDateStr;

  return {
    id: getText(doc.ID),
    issueDate,
    invoiceType: getText(doc.TypeCode) || '380',
    seller: {
      name: getText(seller.Name),
      address: {
        street: getText(seller.PostalTradeAddress?.LineOne),
        city: getText(seller.PostalTradeAddress?.CityName),
        postalCode: getText(seller.PostalTradeAddress?.PostcodeCode),
        country: getText(seller.PostalTradeAddress?.CountryID),
      },
      vatId: getText(seller.SpecifiedTaxRegistration?.ID),
      taxScheme: 'VAT',
    },
    buyer: {
      name: getText(buyer.Name),
      address: {
        street: getText(buyer.PostalTradeAddress?.LineOne),
        city: getText(buyer.PostalTradeAddress?.CityName),
        postalCode: getText(buyer.PostalTradeAddress?.PostcodeCode),
        country: getText(buyer.PostalTradeAddress?.CountryID),
      },
      vatId: getText(buyer.SpecifiedTaxRegistration?.ID),
      taxScheme: 'VAT',
    },
    lines,
    totals: {
      lineTotal: getNumber(monetary.LineTotalAmount),
      allowanceTotal: 0,
      chargeTotal: 0,
      taxTotal: getNumber(monetary.TaxTotalAmount),
      netTotal: getNumber(monetary.TaxBasisTotalAmount),
      grossTotal: getNumber(monetary.GrandTotalAmount),
      amountDue: getNumber(monetary.DuePayableAmount),
    },
    taxes,
    payment: {
      currency: getText(settlement.InvoiceCurrencyCode) || 'EUR',
    },
  };
}

module.exports = { xmlToJSON };
