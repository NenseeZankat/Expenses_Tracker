const mongoose = require("mongoose");

const budgetSchema = new mongoose.Schema(
  {
    userId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "User",
      required: true,
    },
    month: {
      type: String, // Format: "YYYY-MM"
      required: true,
    },
    allocatedBudget: {
      type: Number,
      required: true,
    },
    totalExpenses: {
      type: Number,
      default: 0,
    },
    leftoverAmount: {
      type: Number,
      default: 0,
    },
  },
  {
    timestamps: true,
  }
);

module.exports = mongoose.model("Budget", budgetSchema);
