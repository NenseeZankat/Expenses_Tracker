const mongoose = require("mongoose");

const debtSchema = new mongoose.Schema({
  userId: { type: mongoose.Schema.Types.ObjectId, ref: "User", required: true },
  loanPurpose: { type: String, required: true }, // Purpose of the loan (e.g., Car, House, Education)
  amount: { type: Number, required: true }, // Loan Amount
  interestRate: { type: Number, required: true }, // Annual Interest Rate (%)
  monthlyEMI: { type: Number, required: true }, // Monthly EMI
  totalInterestPaid: { type: Number, default: 0 }, // Total Interest Paid
  remainingAmount: { type: Number, required: true }, // Remaining Balance
  startDate: { type: Date, default: Date.now }, // Loan Start Date
  status: { type: String, enum: ["Active", "Paid"], default: "Active" }, // ✅ New Field
});

const Debt = mongoose.model("Debt", debtSchema);
module.exports = Debt;
