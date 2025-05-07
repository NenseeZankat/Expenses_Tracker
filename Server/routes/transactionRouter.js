const express = require("express");
const usersController = require("../controllers/usersCtrl");
const isAuthenticated = require("../middlewares/isAuth");
const categoryController = require("../controllers/categoryCtrl");
const transactionController = require("../controllers/transactionCtrl");
const Transaction = require("../model/Transaction");

const transactionRouter = express.Router();

//!add
transactionRouter.post(
  "/api/v1/transactions/create",
  isAuthenticated,
  transactionController.create
);
//! lists
transactionRouter.get(
  "/api/v1/transactions/lists",
  isAuthenticated,
  transactionController.getFilteredTransactions
);
//! update
transactionRouter.put(
  "/api/v1/transactions/update/:id",
  isAuthenticated,
  transactionController.update
);
//! delete
transactionRouter.delete(
  "/api/v1/transactions/delete/:id",
  isAuthenticated,
  transactionController.delete
);
transactionRouter.get(
  "/api/v1/transactions/user/:userId",
  
  async (req, res) => {
    console.log(req);
    console.log(res);
    try {
      const transactions = await Transaction.find({ user: req.params.userId });
      res.json(transactions);
    } catch (error) {
      res.status(500).json({ error: "Failed to fetch transactions" });
    }
  }
);



module.exports = transactionRouter;
