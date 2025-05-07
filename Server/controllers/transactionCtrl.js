const asyncHandler = require("express-async-handler");
const Category = require("../model/Category");
const Transaction = require("../model/Transaction");
const User = require("../model/User");
const FamilyTransaction = require("../model/FamilyTransaction");
const mongoose = require("mongoose");
const Budget = require("../model/Budget");

console.log("👉 Reached transactionController.delete");
const transactionController = {
  
 
 

   create :asyncHandler(async (req, res) => {
      const { type, category, amount, paymentMethod, date, description, user, splitExpense, splitUsers } = req.body;
  
      // ✅ Step 1: Validate required fields
      if (!amount || !type || !date) {
          return res.status(400).json({ message: "Type, amount, and date are required" });
      }
  
      // ✅ Step 2: Find the requesting user
      const foundUser = await User.findById(req.user);
      if (!foundUser) {
          return res.status(404).json({ message: "User not found" });
      }


      const month = new Date(date).toISOString().slice(0, 7); // "YYYY-MM" format
      let budget = await Budget.findOne({ userId: req.user, month });
      console.log("Searching for budget with:", { userId: req.user, month });
      console.log("Budget found:", budget);

      let budgetExceededWarning = null;
      if (type === "expense" && budget) {
          const updatedExpense = budget.totalExpenses + amount;
          if (updatedExpense > budget.allocatedBudget) {
              const exceededAmount = updatedExpense - budget.allocatedBudget;
              budgetExceededWarning = `⚠ Warning: Your budget for ${month} is exceeded by ${exceededAmount}!`;
          }
      }
      // ✅ Step 3: Prepare transaction data
      let transactionData = {
          user: user || req.user,  // Use `req.body.user` if provided, else `req.user`
          familyId: foundUser.familyId || null, // Store family ID if exists
          type,
          category,
          amount,
          paymentMethod,
          description,
          date,
          splitExpense: !!splitExpense, // Ensure boolean type
          splitUsers: [],
      };
      console.log("Transaction object:", transactionData); // Check if this is an object, not a string
      // ✅ Step 4: Handle Split Expenses
      if (splitExpense && splitUsers?.length > 0) {
          const splitAmount = (amount / (splitUsers.length + 1)).toFixed(2); // Divide among users including self
  
          transactionData.splitUsers = splitUsers.map(userId => ({
              userId,
              amount: splitAmount,
          }));
      }
  
      // ✅ Step 5: Save the transaction
      const newTransaction = new Transaction(transactionData);
      await newTransaction.save();

      if (type === "expense" && budget) {
        budget.totalExpenses += amount;
        budget.leftoverAmount = budget.allocatedBudget - budget.totalExpenses;
        await budget.save();
      }
      // ✅ Step 6: Update Family Transactions if user is part of a family

      let familyTransaction = await FamilyTransaction.findById(foundUser.familyId);

        if (!familyTransaction) {
            console.log("⚠️ No FamilyTransaction found for familyId:", foundUser.familyId);

            // Create a new family transaction record
            familyTransaction = new FamilyTransaction({
                _id: new mongoose.Types.ObjectId(foundUser.familyId), // Ensure ObjectId
                totalIncome: type === "income" ? amount : 0,
                totalExpense: type === "expense" ? amount : 0,
                transactions: [
                    {
                        user: new mongoose.Types.ObjectId(req.user), // Ensure ObjectId
                        username: foundUser.username,
                        type,
                        category,
                        amount,
                        date: new Date(date), // Ensure Date format
                    }
                ]
            });

            await familyTransaction.save();
        } else {
            // Update existing family transaction record
            await FamilyTransaction.findOneAndUpdate(
                { _id: new mongoose.Types.ObjectId(foundUser.familyId) },
                {
                    $inc: { [type === "income" ? "totalIncome" : "totalExpense"]: amount },
                    $push: {
                        transactions: {
                            user: new mongoose.Types.ObjectId(req.user), // Ensure ObjectId
                            username: foundUser.username,
                            type,
                            category,
                            amount,
                            date: new Date(date), // Ensure Date format
                        }
                    }
                },
                { new: true }
            );
        }
        const responseData = { transaction: newTransaction };
      if (budgetExceededWarning) {
          responseData.warning = budgetExceededWarning;
      }
  
      // ✅ Step 7: Send response
      res.status(201).json(responseData);
  }),
  
  
  
  //!lists
  getFilteredTransactions: asyncHandler(async (req, res) => {
    const { startDate, endDate, type, category } = req.query;
    let filters = { user: req.user };

    if (startDate) {
      filters.date = { ...filters.date, $gte: new Date(startDate) };
    }
    if (endDate) {
      filters.date = { ...filters.date, $lte: new Date(endDate) };
    }
    if (type) {
      filters.type = type;
    }
    if (category) {
      if (category === "All") {
        //!  No category filter needed when filtering for 'All'
      } else if (category === "Uncategorized") {
        //! Filter for transactions that are specifically categorized as 'Uncategorized'
        filters.category = "Uncategorized";
      } else {
        filters.category = category;
      }
    }
    const transactions = await Transaction.find(filters).sort({ date: -1 });
    res.json(transactions);
  }),

  //!update
  update: asyncHandler(async (req, res) => {
    //! Find the transaction
    const transaction = await Transaction.findById(req.params.id);
    if (transaction && transaction.user.toString() === req.user.toString()) {
      (transaction.type = req.body.type || transaction.type),
        (transaction.category = req.body.category || transaction.category),
        (transaction.amount = req.body.amount || transaction.amount),
        (transaction.date = req.body.date || transaction.date),
        (transaction.paymentMethod= req.body.paymentMethod || transaction.paymentMethod),
        (transaction.description =
          req.body.description || transaction.description);
      //update
      const updatedTransaction = await transaction.save();
      res.json(updatedTransaction);
    }
  }),
  //! delete
  delete: asyncHandler(async (req, res) => {
    //! Find the transaction
    console.log("👉 Reached transactionController.delete");
    //console.log(req);
    const transaction = await Transaction.findById(req.params.id);
    if (transaction && transaction.user.toString() === req.user.toString()) {
      await Transaction.findByIdAndDelete(req.params.id);
      res.json({ message: "Transaction removed" });
    }
  }),
};

module.exports = transactionController;
