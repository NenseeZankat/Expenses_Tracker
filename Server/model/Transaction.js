const mongoose = require("mongoose");

const transactionSchema = new mongoose.Schema(
  {
    user: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "User",
      required: true,
    },
    familyId: { type: mongoose.Schema.Types.ObjectId, ref: "Family"
      
     },
    type: {
      type: String,
      required: true,
      enum: ["income", "expense"],
    },
    category: {
      type: String,
      required: true,
      default: "Uncategorized",
    },
    amount: {
      type: Number,
      required: true,
    },
    paymentMethod: {
      type: String,
      required: true,
      enum: ["Cash", "UPI", "Credit Card", "Debit Card", "Bank Transfer", "Other"],
      default: "Cash",
    },
    date: {
      type: Date,
      default: Date.now,
    },
    description: {
      type: String,
      required: false,
    },
    splitExpense: { type: Boolean, default: false },
    splitUsers: [
        {
            userId: { type: mongoose.Schema.Types.ObjectId, ref: "User" },
            amount: { type: Number },
        },
    ],
  },
  {
    timestamps: true,
  }
);

module.exports = mongoose.model("Transaction", transactionSchema);
