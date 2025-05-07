const Transaction = require("../model/Transaction");
const User = require("../model/User"); // ✅ Make sure to import User model
require("dotenv").config();
const stripe = require("stripe")(process.env.STRIPE_SECRET_KEY);

exports.createPaymentIntent = async (req, res) => {
  const { amount, type, category, placeName,user,description,paymentMethod,date } = req.body;


  try {
    console.log("🟢 Received payment data:", { amount, type, category, placeName, user,description,paymentMethod,date });
    
    // ✅ Validate input
    if (!amount || !user) {
      console.error("🔴 Missing required fields");
      return res.status(400).json({ message: "Amount and user are required" });
    }

    // ✅ Find the user using decoded JWT data
    const foundUser = await User.findById(user);
    if (!foundUser) {
      return res.status(404).json({ message: "User not found" });
    }

    // ✅ Create payment intent
    const paymentIntent = await stripe.paymentIntents.create({
      amount: amount * 100, // Convert to cents
      currency: "usd",
    });

    console.log("🟢 Payment intent created:", paymentIntent);

    // ✅ Create a new transaction
    const transaction = new Transaction({
      user: user,
      amount,
      type,
      category,
      paymentMethod: "Credit Card",
      placeName,
      description: `Payment of $${amount} made via card`,
      familyId: foundUser.familyId || null,
      date: date, // ✅ Make sure to provide date
      splitExpense: false, // ✅ Set a default value or get it from req.body
      splitUsers: [],
    });

    await transaction.save();

    console.log("🟢 Transaction saved:", transaction);
    console.log(paymentIntent.client_secret);
    res.status(200).json({
      clientSecret: paymentIntent.client_secret,
      transaction,
    });
  } catch (error) {
    console.error("🔴   Payment intent creation failed:", error.message);
    res.status(500).json({
      message: "Failed to create payment intent",
      error: error.message,
    });
  }
};
