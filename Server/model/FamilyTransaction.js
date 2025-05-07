const mongoose = require("mongoose");

const familyTransactionSchema = new mongoose.Schema({
  _id: { type: mongoose.Schema.Types.ObjectId }, // Use familyId as document ID
  totalIncome: { type: Number, default: 0 },
  totalExpense: { type: Number, default: 0 },
  transactions: [{
    user: { type: mongoose.Schema.Types.ObjectId, ref: "User" },
    username: String,
    type: { type: String, enum: ["income", "expense"] }, // Ensure enum is enforced
    category: String,
    amount: Number,
    paymentMethod: { type: String, enum: ["Cash", "UPI", "Credit Card", "Debit Card", "Bank Transfer", "Other"] },
    date: { type: Date, default: Date.now },
    description: String,
    splitExpense: { type: Boolean, default: false },
    splitUsers: [
      {
        userId: { type: mongoose.Schema.Types.ObjectId, ref: "User" },
        amount: Number,
      },
    ],
  }]
}, { timestamps: true });

module.exports = mongoose.model("FamilyTransaction", familyTransactionSchema);
