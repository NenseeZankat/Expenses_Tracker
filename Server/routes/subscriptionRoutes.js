const express = require("express");
const { addSubscription, getSubscriptions, disableNotification } = require("../controllers/subscriptionCtrl");
const isAuth = require("../middlewares/isAuth");
const subscriptionController = require('../controllers/subscriptionCtrl');

const router = express.Router();

router.post("/add", isAuth, addSubscription);
router.get("/:userId", isAuth, getSubscriptions);
router.put("/:id/disable", isAuth, disableNotification);
router.put("/:id/enable",isAuth, subscriptionController.enableNotification);
router.delete("/:id",isAuth, subscriptionController.deleteSubscription);
module.exports = router;
