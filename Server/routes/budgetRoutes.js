const express = require("express");
const budgetController = require("../controllers/budgetController");
const isAuthenticated = require("../middlewares/isAuth");

const budgetRouter = express.Router();

//! Add Budget
budgetRouter.post(
  "/create",
  isAuthenticated,
  budgetController.createBudget
);

//! Get Budget (by month)
budgetRouter.get(
  "/lists",
  isAuthenticated,
  budgetController.getBudget
);

//! Update Budget (on transaction)
budgetRouter.put(
  "/updatetransaction",
  isAuthenticated,
  budgetController.updateBudgetOnTransaction
);

budgetRouter.put(
    "/update",
    isAuthenticated,
    budgetController.updateBudget 
  );
module.exports = budgetRouter;
