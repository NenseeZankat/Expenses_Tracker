const asyncHandler = require("express-async-handler");
const Budget = require("../model/Budget");
const Transaction = require("../model/Transaction");


const createBudget = asyncHandler(async (req, res) => {
    const { month, allocatedBudget } = req.body;
  
    if (!month || !allocatedBudget) {
      res.status(400);
      throw new Error("Month and budget are required.");
    }
    const existingBudget = await Budget.findOne({ userId: req.user, month });

    if (existingBudget) {
      return res.status(400).json({ error: `A budget for ${month} already exists.` });
    }
  
    // Convert `month` (YYYY-MM) into a date range
    const startDate = new Date(`${month}-01`); // First day of the month
    const endDate = new Date(startDate);
    endDate.setMonth(endDate.getMonth() + 1); // Move to the first day of next month
  
    // Fetch all expense transactions for the given month
    const transactions = await Transaction.find({
      user: req.user,
      date: { $gte: startDate, $lt: endDate }, // Filter between start and end of month
      type: "expense",
    });
  
    // Calculate total expenses
    const totalExpenses = transactions.reduce((sum, txn) => sum + txn.amount, 0);
  
    // Calculate leftover amount
     const leftoverAmount = allocatedBudget - totalExpenses;
    // if ( leftoverAmount < 0) {
    //   return res.status(400).json({
    //     error: `Your expenses exceed your budget by ${Math.abs(leftoverAmount)} units.`,
    //   });
    // }
    // Create budget with calculated values
    const budget = new Budget({
      userId: req.user,
      month,
      allocatedBudget,
      totalExpenses,
      leftoverAmount,
    });
  
    await budget.save();
    res.status(201).json(budget);
  });
  


  const updateBudget = asyncHandler(async (req, res) => {
    const { month, amount } = req.body;
  
    if (!month || !amount) {
      res.status(400);
      throw new Error("Month and new budget amount are required.");
    }
    // console.log("requser",req.user);
  
    // let budget = await Budget.findOne({ userId: req.user, month });


    console.log("Finding budget with:", { userId: req.user, month });

    let budget = await Budget.findOne({ userId: req.user, month });

if (!budget) {
  console.log("Budget not found in database!");
  res.status(404);
  throw new Error("No budget found for this month.");
}

    if (!budget) {
      res.status(404);
      throw new Error("No budget found for this month.");
    }
  
    // Update allocated budget and recalculate leftover amount
    budget.allocatedBudget = amount;
    budget.leftoverAmount = amount - budget.totalExpenses;
    
    
    await budget.save();
    res.status(201).json(budget);
  });

// ✅ Get the budget for a user (filter by month)
const getBudget = asyncHandler(async (req, res) => {
  const { month } = req.query;
  console.log("req user",req.user);
  console.log(month)
  const budget = await Budget.findOne({ userId: req.user, month });

  if (!budget) {
    res.status(404);
    throw new Error("Budget not found for the selected month.");
  }

  res.json(budget);
});

// ✅ Update the budget when a new transaction is added
const updateBudgetOnTransaction = asyncHandler(async (req, res) => {
  const { amount, type, date } = req.body;
  const month = date.substring(0, 7); // Extract "YYYY-MM"

  let budget = await Budget.findOne({ userId: req.user, month });

  if (!budget) {
    res.status(400);
    throw new Error("No budget set for this month.");
  }

  if (type === "expense") {
    budget.totalExpenses += amount;
    budget.leftoverAmount = budget.allocatedBudget - budget.totalExpenses;
  }

  await budget.save();
  res.json(budget);
});

module.exports = { createBudget, getBudget, updateBudgetOnTransaction,updateBudget  };
