// backend/routes/paymentRouter.js
const express = require("express");
const isAuthenticated = require("../middlewares/isAuth");
const paymentController = require("../controllers/paymentController");

const router = express.Router();

router.post("/create", paymentController.createPaymentIntent);
// router.post(
//     "/create",
//     isAuthenticated,
//     paymentController.createPaymentIntent
//   );
module.exports = router;
