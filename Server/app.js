const express = require("express");
const mongoose = require("mongoose");
const cors = require("cors");
const userRouter = require("./routes/userRouter");
const errorHandler = require("./middlewares/errorHandlerMiddleware");
const categoryRouter = require("./routes/categoryRouter");
const transactionRouter = require("./routes/transactionRouter");
const reportRoutes = require("./routes/reportRoutes");
const familyrouters = require("./routes/familyRoutes");
const budgetrouters = require("./routes/budgetRoutes");
const schedule = require("node-schedule");
const Budget = require("./model/Budget");
const debtrouters = require("./routes/debtRoutes");
const familyGoalRoutes = require("./routes/familyGoalRoutes");
const paymentRoutes = require("./routes/paymentRoutes");
// require("./middlewares/realTimeSync"); // Import real-time sync
// const voicerouters = require("./routes/voiceRoutes");
const familyTransactionRoutes = require("./routes/familyTransactionRoutes");
// const cron = require("./middlewares/budgetScheduler");
const subscriptionrouter=require("./routes/subscriptionRoutes")
const path = require("path");
const app = express();
const userRoutes = require("./routes/usersautomation");
require("./middlewares/budgetScheduler");
require("./cronJobs");

const stripe = require('stripe')(process.env.STRIPE_SECRET_KEY);

console.log('Stripe Secret Key:', process.env.STRIPE_SECRET_KEY);
//!Connect to mongodb
// mongoose
//   .connect("mongodb://localhost:27017/Expense_tracker")
//   .then(() => console.log("DB Connected"))
//   .catch((e) => console.log(e));

  mongoose
  .connect("mongodb+srv://desairajvi8:FBkjLZ1JbtXgge0n@expensetrackercluster.irbjt.mongodb.net/?retryWrites=true&w=majority&appName=ExpenseTrackerCluster",{
    useNewUrlParser: true,
    useUnifiedTopology: true,
    
  })
  .then(() => console.log("DB Connected"))
  .catch((e) => console.log(e));
//! Cors config
const corsOptions = {
  origin: ["https://frontend7-6mma.onrender.com"],
};
app.use((req, res, next) => {
  console.log(`🟢 Incoming Request: ${req.method} ${req.url}`);
  next();
});
app.use(cors(corsOptions));
//!Middlewares
app.use(express.json()); //?Pass incoming json data
//!Routes
// app.use("/", voicerouters);
app.use("/", userRouter);
app.use("/", familyrouters);
app.use("/", categoryRouter);
//app.use("/api/v1/transactions", transactionRouter);
app.use("/", transactionRouter);
app.use("/api/subscriptions",subscriptionrouter );
app.use("/api/user", userRoutes);
app.use("/api/reports", reportRoutes);
app.use("/api/budgets", budgetrouters);
app.use("/api/debt", debtrouters);
app.use("/api/family-transactions", familyTransactionRoutes);
app.use("/api/family-goals", familyGoalRoutes);
app.use("/uploads", express.static(path.join(__dirname, "uploads")));
app.use("/api/v1/payments", paymentRoutes);


//! Error
app.use(errorHandler);
// cron.start();
//!Start the server
const PORT = process.env.PORT || 8000;
app.listen(PORT, () =>
  console.log(`Server is running on this port... ${PORT} `)
);

// const carryOverBudget = async () => {
//   try {
//     console.log("Running monthly carry-over job...");

//     // Get current date and determine previous month
//     const now = new Date();
//     let year = now.getFullYear();
//     let month = now.getMonth(); // 0-based (0 = Jan, 11 = Dec)

//     // Handle January case (previous month = December of previous year)
//     let prevYear = month === 0 ? year - 1 : year;
//     let prevMonth = month === 0 ? 11 : month - 1;
//     let prevMonthStr = `${prevYear}-${String(prevMonth + 1).padStart(2, "0")}`; // "YYYY-MM"

//     // Get next month in "YYYY-MM" format
//     let nextYear = month === 11 ? year + 1 : year;
//     let nextMonth = month === 11 ? 0 : month;
//     let nextMonthStr = `${nextYear}-${String(nextMonth + 1).padStart(2, "0")}`; // "YYYY-MM"

//     // Find all budgets for the previous month
//     const prevBudgets = await Budget.find({ month: prevMonthStr });

//     for (let prevBudget of prevBudgets) {
//       const { userId, leftoverAmount } = prevBudget;

//       // Check if next month's budget already exists
//       let nextBudget = await Budget.findOne({ userId, month: nextMonthStr });

//       if (!nextBudget) {
//         // If no budget exists, create a new one and carry over the leftover amount
//         nextBudget = new Budget({
//           userId,
//           month: nextMonthStr,
//           allocatedBudget: leftoverAmount, // Carry over leftover
//           totalExpenses: 0,
//           leftoverAmount: leftoverAmount, // Initially set as leftover
//         });

//         await nextBudget.save();
//         console.log(`Carried over ${leftoverAmount} for user ${userId} to ${nextMonthStr}`);
//       } else {
//         // If a budget exists, update it by adding leftoverAmount
//         nextBudget.allocatedBudget += leftoverAmount;
//         nextBudget.leftoverAmount += leftoverAmount;
//         await nextBudget.save();
//         console.log(`Updated budget for user ${userId} in ${nextMonthStr} with carryover.`);
//       }
//     }

//     console.log("✅ Monthly carry-over process completed.");
//   } catch (error) {
//     console.error("❌ Error during budget carry-over:", error);
//   }
// };

// //! 📅 Schedule Job to Run at 23:59 on the Last Day of Every Month
// // schedule.scheduleJob("59 23 28-31 * *", async () => {
// //   const today = new Date();
// //   const tomorrow = new Date(today);
// //   tomorrow.setDate(today.getDate() + 1);

// //   if (tomorrow.getMonth() !== today.getMonth()) {
// //     console.log("📅 Last day of the month detected. Running carryOverBudget...");
// //     await carryOverBudget();
// //   } else {
// //     console.log("🚫 Not the last day. Skipping carryOverBudget.");
// //   }
// // });
// carryOverBudget()
//   .then(() => console.log("✅ Manual Test Success!"))
//   .catch(err => console.error("❌ Error:", err));


// //! Start Server
// const PORT = process.env.PORT || 8000;
// app.listen(PORT, () => console.log(`🚀 Server is running on port ${PORT}`));

