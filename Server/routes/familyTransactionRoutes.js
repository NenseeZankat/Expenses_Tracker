const express = require("express");
const { getFamilyTransactions } = require("../controllers/familyTransactionController");

const router = express.Router();

router.get("/:familyId", getFamilyTransactions); // Get family transactions

module.exports = router;
