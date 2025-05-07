const mongoose = require("mongoose");

const subscriptionSchema = new mongoose.Schema({
  userId: { type: mongoose.Schema.Types.ObjectId, ref: "User", required: true },
  serviceName: { type: String, required: true },
  amount: { type: Number, required: true },
  renewalDate: { type: Date, required: true },
  frequency: { type: String, enum: ["monthly", "yearly", "weekly"], required: true }, 
  alertDaysBefore: { type: Number, default: 3 }, // Notify 3 days before renewal
  notified: { type: Boolean, default: false }, // To track if notification is sent
  notify: { type: Boolean, default: true }
});

module.exports = mongoose.model("Subscription", subscriptionSchema);
