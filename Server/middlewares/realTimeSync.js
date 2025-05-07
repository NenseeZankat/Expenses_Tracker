// const mongoose = require("mongoose");
// const Transaction = require("../model/Transaction");
// const User = require("../model/User");
// const FamilyTransaction = require("../model/FamilyTransaction");

// // Listen for real-time changes in the Transaction collection
// const transactionStream = Transaction.watch();

// transactionStream.on("change", async (change) => {
//   if (change.operationType === "insert") {
//     const newTransaction = change.fullDocument;

//     // Fetch user details to get familyId
//     const user = await User.findById(newTransaction.user);
//     if (!user || !user.familyId) return;

//     const { familyId, username } = user;
//     const { type, amount, category, date } = newTransaction;

//     // Update family transactions
//     await FamilyTransaction.findOneAndUpdate(
//       { _id: familyId },
//       {
//         $inc: { [type === "income" ? "totalIncome" : "totalExpense"]: amount },
//         $push: {
//           transactions: {
//             user: newTransaction.user,
//             username,
//             type,
//             category,
//             amount,
//             date
//           }
//         }
//       },
//       { upsert: true, new: true }
//     );

//     console.log(`Updated family transactions for familyId: ${familyId}`);
//   }
// });

// console.log("Listening for real-time transaction updates...");
