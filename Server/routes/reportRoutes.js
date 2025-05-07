const express = require("express");
const { generateTaxReport } = require("../controllers/reportCtrl");
const router = express.Router();

router.get("/generate/:userId/:year", generateTaxReport);


module.exports = router;
