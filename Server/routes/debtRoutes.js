const express = require("express");
const { addDebt, getDebt, makePayment, deleteDebt } = require("../controllers/debtController");
const isAuthenticated = require("../middlewares/isAuth");
const router = express.Router();

router.post("/add", isAuthenticated, addDebt); // Add New Debt
router.get("/get", isAuthenticated, getDebt); // Fetch User's Debt
router.post("/pay", isAuthenticated, makePayment); // Make Payment
router.delete("/delete/:debtId", isAuthenticated, deleteDebt);
module.exports = router;
