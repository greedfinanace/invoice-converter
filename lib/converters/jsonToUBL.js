const { create } = require('xmlbuilder2');
const Decimal = require('decimal.js');

async function jsonToUBL(invoice, ciusCountry = 'DE') {
  const root = create({ version: '1.0', encoding: 'UTF-8' })
    .ele('Invoice', {
      xmlns: 'urn:oasis:names:specification:ubl:schema:xsd:Invoice-2',
      'xmlns:cac': 'urn:oasis:names:specification:ubl:schema:xsd:CommonAggregateComponents-2',
      'xmlns:cbc': 'urn:oasis:names:specification:ubl:schema:xsd:CommonBasicComponents-2',
    });

  const currency = invoice.payment?.currency || 'EUR';

  // Invoice metadata
  root.ele('cbc:CustomizationID').txt('urn:cen.eu:en16931:2017#compliant#urn:fdc:peppol.eu:2017:poacc:billing:3.0');
  root.ele('cbc:ProfileID').txt('urn:fdc:peppol.eu:2017:poacc:billing:01:1.0');
  root.ele('cbc:ID').txt(invoice.id);
  root.ele('cbc:IssueDate').txt(invoice.issueDate);
  root.ele('cbc:InvoiceTypeCode').txt(invoice.invoiceType || '380');

  if (invoice.dueDate) {
    root.ele('cbc:DueDate').txt(invoice.dueDate);
  }

  root.ele('cbc:DocumentCurrencyCode').txt(currency);

  if (invoice.notes) {
    root.ele('cbc:Note').txt(invoice.notes);
  }

  // Seller
  const supplier = root.ele('cac:AccountingSupplierParty');
  const supplierParty = supplier.ele('cac:Party');
  
  if (invoice.seller.endpoint) {
    supplierParty.ele('cbc:EndpointID', { schemeID: 'EM' }).txt(invoice.seller.endpoint);
  }
  
  const supplierName = supplierParty.ele('cac:PartyName');
  supplierName.ele('cbc:Name').txt(invoice.seller.name);

  const supplierAddress = supplierParty.ele('cac:PostalAddress');
  supplierAddress.ele('cbc:StreetName').txt(invoice.seller.address.street || '');
  supplierAddress.ele('cbc:CityName').txt(invoice.seller.address.city || '');
  supplierAddress.ele('cbc:PostalZone').txt(invoice.seller.address.postalCode || '');
  const supplierCountry = supplierAddress.ele('cac:Country');
  supplierCountry.ele('cbc:IdentificationCode').txt(invoice.seller.address.country);

  const supplierTax = supplierParty.ele('cac:PartyTaxScheme');
  supplierTax.ele('cbc:CompanyID').txt(invoice.seller.vatId);
  const supplierTaxScheme = supplierTax.ele('cac:TaxScheme');
  supplierTaxScheme.ele('cbc:ID').txt(invoice.seller.taxScheme || 'VAT');

  const supplierLegal = supplierParty.ele('cac:PartyLegalEntity');
  supplierLegal.ele('cbc:RegistrationName').txt(invoice.seller.name);

  // Buyer
  const customer = root.ele('cac:AccountingCustomerParty');
  const customerParty = customer.ele('cac:Party');
  
  if (invoice.buyer.endpoint) {
    customerParty.ele('cbc:EndpointID', { schemeID: 'EM' }).txt(invoice.buyer.endpoint);
  }

  const customerName = customerParty.ele('cac:PartyName');
  customerName.ele('cbc:Name').txt(invoice.buyer.name);

  const customerAddress = customerParty.ele('cac:PostalAddress');
  customerAddress.ele('cbc:StreetName').txt(invoice.buyer.address.street || '');
  customerAddress.ele('cbc:CityName').txt(invoice.buyer.address.city || '');
  customerAddress.ele('cbc:PostalZone').txt(invoice.buyer.address.postalCode || '');
  const customerCountry = customerAddress.ele('cac:Country');
  customerCountry.ele('cbc:IdentificationCode').txt(invoice.buyer.address.country);

  if (invoice.buyer.vatId) {
    const customerTax = customerParty.ele('cac:PartyTaxScheme');
    customerTax.ele('cbc:CompanyID').txt(invoice.buyer.vatId);
    const customerTaxScheme = customerTax.ele('cac:TaxScheme');
    customerTaxScheme.ele('cbc:ID').txt(invoice.buyer.taxScheme || 'VAT');
  }

  const customerLegal = customerParty.ele('cac:PartyLegalEntity');
  customerLegal.ele('cbc:RegistrationName').txt(invoice.buyer.name);

  // Payment Means
  if (invoice.payment?.iban) {
    const paymentMeans = root.ele('cac:PaymentMeans');
    paymentMeans.ele('cbc:PaymentMeansCode').txt('30');
    const paymentAccount = paymentMeans.ele('cac:PayeeFinancialAccount');
    paymentAccount.ele('cbc:ID').txt(invoice.payment.iban);
    if (invoice.payment.bic) {
      const bank = paymentAccount.ele('cac:FinancialInstitutionBranch');
      bank.ele('cbc:ID').txt(invoice.payment.bic);
    }
  }

  // Tax Total
  const taxTotal = root.ele('cac:TaxTotal');
  taxTotal.ele('cbc:TaxAmount', { currencyID: currency })
    .txt(new Decimal(invoice.totals.taxTotal).toFixed(2));

  // Tax subtotals by category
  const taxByCategory = {};
  for (const line of invoice.lines) {
    const cat = line.tax.category;
    if (!taxByCategory[cat]) {
      taxByCategory[cat] = { taxable: 0, tax: 0, rate: line.tax.rate };
    }
    taxByCategory[cat].taxable += line.lineTotal;
    taxByCategory[cat].tax += line.tax.amount;
  }

  for (const [category, amounts] of Object.entries(taxByCategory)) {
    const taxSubtotal = taxTotal.ele('cac:TaxSubtotal');
    taxSubtotal.ele('cbc:TaxableAmount', { currencyID: currency })
      .txt(new Decimal(amounts.taxable).toFixed(2));
    taxSubtotal.ele('cbc:TaxAmount', { currencyID: currency })
      .txt(new Decimal(amounts.tax).toFixed(2));
    
    const taxCat = taxSubtotal.ele('cac:TaxCategory');
    taxCat.ele('cbc:ID').txt(category);
    taxCat.ele('cbc:Percent').txt(amounts.rate.toString());
    const taxCatScheme = taxCat.ele('cac:TaxScheme');
    taxCatScheme.ele('cbc:ID').txt('VAT');
  }

  // Monetary Totals
  const monetary = root.ele('cac:LegalMonetaryTotal');
  monetary.ele('cbc:LineExtensionAmount', { currencyID: currency })
    .txt(new Decimal(invoice.totals.lineTotal).toFixed(2));
  monetary.ele('cbc:TaxExclusiveAmount', { currencyID: currency })
    .txt(new Decimal(invoice.totals.netTotal).toFixed(2));
  monetary.ele('cbc:TaxInclusiveAmount', { currencyID: currency })
    .txt(new Decimal(invoice.totals.grossTotal).toFixed(2));
  monetary.ele('cbc:PayableAmount', { currencyID: currency })
    .txt(new Decimal(invoice.totals.amountDue).toFixed(2));

  // Invoice Lines
  for (const line of invoice.lines) {
    const invoiceLine = root.ele('cac:InvoiceLine');
    invoiceLine.ele('cbc:ID').txt(line.id);
    invoiceLine.ele('cbc:InvoicedQuantity', { unitCode: line.unit || 'EA' }).txt(line.quantity.toString());
    invoiceLine.ele('cbc:LineExtensionAmount', { currencyID: currency })
      .txt(new Decimal(line.lineTotal).toFixed(2));

    const item = invoiceLine.ele('cac:Item');
    item.ele('cbc:Name').txt(line.description);
    
    const classifiedTax = item.ele('cac:ClassifiedTaxCategory');
    classifiedTax.ele('cbc:ID').txt(line.tax.category);
    classifiedTax.ele('cbc:Percent').txt(line.tax.rate.toString());
    const classifiedTaxScheme = classifiedTax.ele('cac:TaxScheme');
    classifiedTaxScheme.ele('cbc:ID').txt('VAT');

    const price = invoiceLine.ele('cac:Price');
    price.ele('cbc:PriceAmount', { currencyID: currency })
      .txt(new Decimal(line.unitPrice).toFixed(2));
  }

  return root.end({ prettyPrint: true });
}

module.exports = { jsonToUBL };
