const FamilyGoal = require("../model/FamilyGoal");
const Transaction = require("../model/Transaction"); // Import Transaction model
const FamilyTransaction = require("../model/FamilyTransaction");
const User = require("../model/User");
// ✅ Create a new Family Goal (Expense Challenge)
exports.createFamilyGoal = async (req, res) => {
  try {
    console.log("Request Body:", req.body); // Debug input data

    const { familyId, goalName, targetAmount, startDate, endDate, reward } = req.body;

    const newGoal = new FamilyGoal({
      familyId,
      goalName,
      targetAmount,
      startDate,
      endDate,
      reward,
    });

    await newGoal.save();
    res.status(201).json(newGoal);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
};

// ✅ Get all Family Goals for a specific family
exports.getFamilyGoals = async (req, res) => {
  try {
    const goals = await FamilyGoal.find({ familyId: req.params.familyId });
    res.status(200).json(goals);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
};

// ✅ Join an Expense Challenge
exports.joinFamilyGoal = async (req, res) => {
  try {
    const { userId } = req.body;
    const goal = await FamilyGoal.findById(req.params.goalId);

    if (!goal) return res.status(404).json({ message: "Goal not found" });

    goal.participants.push({ userId, contribution: 0 });
    await goal.save();

    res.json(goal);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
};

// ✅ Update Contribution for an Expense Challenge
exports.updateContribution = async (req, res) => {
  try {
    const { userId, amount } = req.body;
    const user = await User.findById(userId).select("username"); // 🔹 Fetch name
    console.log(user);
    if (!user) return res.status(404).json({ message: "User not found" });
    
    const goal = await FamilyGoal.findById(req.params.goalId);

    if (!goal) return res.status(404).json({ message: "Goal not found" });

    let participant = goal.participants.find((p) => p.userId.toString() === userId);
    if (!participant) return res.status(400).json({ message: "User not in challenge" });

    participant.contribution += amount;
    goal.currentAmount += amount;

    // Check if goal is completed
    if (goal.currentAmount >= goal.targetAmount) {
      goal.status = "Completed";
    }


    const transaction = new Transaction({
      user: userId,
      familyId: goal.familyId, // Link to family goal
      type: "expense", // Since it's a contribution (spending)
      category: "Family Goal Contribution",
      amount,
      paymentMethod: "UPI", // Default payment method (can be changed)
      date: new Date(),
      description: `Contributed ₹${amount} to ${goal.goalName}`,
    });

    await transaction.save();
    
    
    let familyTransaction = await FamilyTransaction.findById(goal.familyId._id);

    if (!familyTransaction) {
      // Create new family transaction entry if it doesn't exist
      familyTransaction = new FamilyTransaction({
        _id: goal.familyId._id,
        totalExpense: amount,
        transactions: [
          {
            user: userId,
            username:user.username, // TODO: Fetch user name from DB
            type: "expense",
            category: "Family Goal Contribution",
            amount,
            paymentMethod: "UPI",
            date: new Date(),
            description: `Contributed ₹${amount} to ${goal.goalName}`,
          },
        ],
      });
    } else {
      // Update existing family transaction
      familyTransaction.totalExpense += amount;
      familyTransaction.transactions.push({
        user: userId,
        username: user.username, // TODO: Fetch user name from DB
        type: "expense",
        category: "Family Goal Contribution",
        amount,
        paymentMethod: "UPI",
        date: new Date(),
        description: `Contributed ₹${amount} to ${goal.goalName}`,
      });
    }

    await familyTransaction.save(); // Save family transaction// Save transaction

    await goal.save();
    res.json(goal);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
};

// ✅ Delete a Family Goal
exports.deleteFamilyGoal = async (req, res) => {
  try {
    await FamilyGoal.findByIdAndDelete(req.params.goalId);
    res.json({ message: "Goal deleted successfully" });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
};
