const { create } = require('xmlbuilder2');
const Decimal = require('decimal.js');

async function jsonToCII(invoice, ciusCountry = 'DE') {
  const root = create({ version: '1.0', encoding: 'UTF-8' })
    .ele('rsm:CrossIndustryInvoice', {
      'xmlns:rsm': 'urn:un:unece:uncefact:data:standard:CrossIndustryInvoice:100',
      'xmlns:ram': 'urn:un:unece:uncefact:data:standard:ReusableAggregateBusinessInformationEntity:100',
      'xmlns:qdt': 'urn:un:unece:uncefact:data:standard:QualifiedDataType:100',
      'xmlns:udt': 'urn:un:unece:uncefact:data:standard:UnqualifiedDataType:100',
    });

  const currency = invoice.payment?.currency || 'EUR';

  // Exchanged Document Context
  const context = root.ele('rsm:ExchangedDocumentContext');
  const guideline = context.ele('ram:GuidelineSpecifiedDocumentContextParameter');
  guideline.ele('ram:ID').txt('urn:cen.eu:en16931:2017');

  // Exchanged Document
  const document = root.ele('rsm:ExchangedDocument');
  document.ele('ram:ID').txt(invoice.id);
  document.ele('ram:TypeCode').txt(invoice.invoiceType || '380');
  const issueDate = document.ele('ram:IssueDateTime');
  const issueDateFormat = issueDate.ele('udt:DateTimeString', { format: '102' });
  issueDateFormat.txt(invoice.issueDate.replace(/-/g, ''));

  if (invoice.notes) {
    const note = document.ele('ram:IncludedNote');
    note.ele('ram:Content').txt(invoice.notes);
  }

  // Supply Chain Trade Transaction
  const transaction = root.ele('rsm:SupplyChainTradeTransaction');

  // Line Items
  for (const line of invoice.lines) {
    const lineItem = transaction.ele('ram:IncludedSupplyChainTradeLineItem');
    
    const lineDoc = lineItem.ele('ram:AssociatedDocumentLineDocument');
    lineDoc.ele('ram:LineID').txt(line.id);

    const product = lineItem.ele('ram:SpecifiedTradeProduct');
    product.ele('ram:Name').txt(line.description);

    const lineAgreement = lineItem.ele('ram:SpecifiedLineTradeAgreement');
    const netPrice = lineAgreement.ele('ram:NetPriceProductTradePrice');
    netPrice.ele('ram:ChargeAmount').txt(new Decimal(line.unitPrice).toFixed(2));

    const lineDelivery = lineItem.ele('ram:SpecifiedLineTradeDelivery');
    lineDelivery.ele('ram:BilledQuantity', { unitCode: line.unit || 'EA' }).txt(line.quantity.toString());

    const lineSettlement = lineItem.ele('ram:SpecifiedLineTradeSettlement');
    const lineTax = lineSettlement.ele('ram:ApplicableTradeTax');
    lineTax.ele('ram:TypeCode').txt('VAT');
    lineTax.ele('ram:CategoryCode').txt(line.tax.category);
    lineTax.ele('ram:RateApplicablePercent').txt(line.tax.rate.toString());

    const lineMonetary = lineSettlement.ele('ram:SpecifiedTradeSettlementLineMonetarySummation');
    lineMonetary.ele('ram:LineTotalAmount').txt(new Decimal(line.lineTotal).toFixed(2));
  }

  // Header Trade Agreement
  const agreement = transaction.ele('ram:ApplicableHeaderTradeAgreement');

  // Seller
  const seller = agreement.ele('ram:SellerTradeParty');
  seller.ele('ram:Name').txt(invoice.seller.name);

  const sellerAddress = seller.ele('ram:PostalTradeAddress');
  sellerAddress.ele('ram:PostcodeCode').txt(invoice.seller.address.postalCode || '');
  sellerAddress.ele('ram:LineOne').txt(invoice.seller.address.street || '');
  sellerAddress.ele('ram:CityName').txt(invoice.seller.address.city || '');
  sellerAddress.ele('ram:CountryID').txt(invoice.seller.address.country);

  const sellerTax = seller.ele('ram:SpecifiedTaxRegistration');
  sellerTax.ele('ram:ID', { schemeID: 'VA' }).txt(invoice.seller.vatId);

  // Buyer
  const buyer = agreement.ele('ram:BuyerTradeParty');
  buyer.ele('ram:Name').txt(invoice.buyer.name);

  const buyerAddress = buyer.ele('ram:PostalTradeAddress');
  buyerAddress.ele('ram:PostcodeCode').txt(invoice.buyer.address.postalCode || '');
  buyerAddress.ele('ram:LineOne').txt(invoice.buyer.address.street || '');
  buyerAddress.ele('ram:CityName').txt(invoice.buyer.address.city || '');
  buyerAddress.ele('ram:CountryID').txt(invoice.buyer.address.country);

  if (invoice.buyer.vatId) {
    const buyerTax = buyer.ele('ram:SpecifiedTaxRegistration');
    buyerTax.ele('ram:ID', { schemeID: 'VA' }).txt(invoice.buyer.vatId);
  }

  // Header Trade Delivery
  const delivery = transaction.ele('ram:ApplicableHeaderTradeDelivery');
  const shipTo = delivery.ele('ram:ShipToTradeParty');
  shipTo.ele('ram:Name').txt(invoice.buyer.name);

  // Header Trade Settlement
  const settlement = transaction.ele('ram:ApplicableHeaderTradeSettlement');
  settlement.ele('ram:InvoiceCurrencyCode').txt(currency);

  // Payment Means
  if (invoice.payment?.iban) {
    const paymentMeans = settlement.ele('ram:SpecifiedTradeSettlementPaymentMeans');
    paymentMeans.ele('ram:TypeCode').txt('30');
    const payeeAccount = paymentMeans.ele('ram:PayeePartyCreditorFinancialAccount');
    payeeAccount.ele('ram:IBANID').txt(invoice.payment.iban);
    if (invoice.payment.bic) {
      const payeeInstitution = paymentMeans.ele('ram:PayeeSpecifiedCreditorFinancialInstitution');
      payeeInstitution.ele('ram:BICID').txt(invoice.payment.bic);
    }
  }

  // Tax breakdown
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
    const tradeTax = settlement.ele('ram:ApplicableTradeTax');
    tradeTax.ele('ram:CalculatedAmount').txt(new Decimal(amounts.tax).toFixed(2));
    tradeTax.ele('ram:TypeCode').txt('VAT');
    tradeTax.ele('ram:BasisAmount').txt(new Decimal(amounts.taxable).toFixed(2));
    tradeTax.ele('ram:CategoryCode').txt(category);
    tradeTax.ele('ram:RateApplicablePercent').txt(amounts.rate.toString());
  }

  // Monetary Summation
  const monetary = settlement.ele('ram:SpecifiedTradeSettlementHeaderMonetarySummation');
  monetary.ele('ram:LineTotalAmount').txt(new Decimal(invoice.totals.lineTotal).toFixed(2));
  monetary.ele('ram:TaxBasisTotalAmount').txt(new Decimal(invoice.totals.netTotal).toFixed(2));
  monetary.ele('ram:TaxTotalAmount', { currencyID: currency }).txt(new Decimal(invoice.totals.taxTotal).toFixed(2));
  monetary.ele('ram:GrandTotalAmount').txt(new Decimal(invoice.totals.grossTotal).toFixed(2));
  monetary.ele('ram:DuePayableAmount').txt(new Decimal(invoice.totals.amountDue).toFixed(2));

  return root.end({ prettyPrint: true });
}

module.exports = { jsonToCII };
