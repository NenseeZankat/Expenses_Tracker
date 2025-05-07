const Subscription = require("../model/Subscription");

// ✅ Add Subscription
exports.addSubscription = async (req, res) => {
  try {
    const { serviceName, amount, renewalDate, frequency, alertDaysBefore } = req.body;
   
    const newSubscription = new Subscription({
      userId:req.user,
      serviceName,
      amount,
      renewalDate,
      frequency,
      alertDaysBefore,
    });

    await newSubscription.save();
    res.status(201).json({ message: "Subscription added successfully" });
  } catch (error) {
    res.status(500).json({ message: "Error adding subscription", error });
  }
};

// ✅ Fetch User's Subscriptions
exports.getSubscriptions = async (req, res) => {
  try {
    const subscriptions = await Subscription.find({ userId: req.params.userId });
    res.json(subscriptions);
  } catch (error) {
    res.status(500).json({ message: "Error fetching subscriptions", error });
  }
};

exports.disableNotification = async (req, res) => {
    try {
        await Subscription.findByIdAndUpdate(req.params.id, { notify: false });
        res.json({ message: "Notifications disabled for this subscription" });
    } catch (error) {
        res.status(500).json({ message: "Error disabling notification", error });
    }
};


exports.enableNotification = async (req, res) => {
    try {
      await Subscription.findByIdAndUpdate(req.params.id, { notify: true });
      res.json({ message: "Notifications enabled for this subscription" });
    } catch (error) {
      res.status(500).json({ message: "Error enabling notification", error });
    }
  };
  
  // 🗑️ Delete Subscription
  exports.deleteSubscription = async (req, res) => {
    try {
      await Subscription.findByIdAndDelete(req.params.id);
      res.json({ message: "Subscription deleted successfully" });
    } catch (error) {
      res.status(500).json({ message: "Error deleting subscription", error });
    }
  };