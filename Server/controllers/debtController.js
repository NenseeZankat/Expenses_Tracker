const Debt = require("../model/Debt");
const mongoose = require("mongoose");
const Transaction = require("../model/Transaction");
const FamilyTransaction = require("../model/FamilyTransaction");
const User = require("../model/User");
// ✅ Add New Debt
exports.addDebt = async (req, res) => {
  try {
    const { amount, loanPurpose, interestRate, monthlyEMI } = req.body;
    const totalInterest = (amount * (interestRate / 100) * 12) / monthlyEMI; // Estimate interest
    const debt = new Debt({
      userId: req.user,
      amount,
      loanPurpose,
      interestRate,
      monthlyEMI,
      totalInterestPaid: 0,
      remainingAmount: amount,
      status : "Active",
    });
    await debt.save();
    res.status(201).json(debt);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};

exports.getDebt = async (req, res) => {
    try {
      const debts = await Debt.find({ userId: req.user }); // Fetch all debts for the user
      if (!debts.length) return res.status(404).json({ message: "No active debts found" });
      res.json(debts);
    } catch (error) {
      res.status(500).json({ message: error.message });
    }
  };
  

exports.makePayment = async (req, res) => {
  try {
    const { debtId, amount } = req.body;
    const userId = req.user; 

    const user = await User.findById(userId).select("username"); // 🔹 Fetch name
    if (!user) return res.status(404).json({ message: "User not found" });

    const userfamily = await User.findById(userId).select("familyId"); // 🔹 Fetch name
    if (!userfamily) return res.status(404).json({ message: "User not found" });
    // ✅ Ensure debtId is a valid ObjectId
    if (!mongoose.Types.ObjectId.isValid(debtId)) {
      return res.status(400).json({ message: "Invalid Debt ID" });
    }

    let debt = await Debt.findById(debtId);

    if (!debt) return res.status(404).json({ message: "No debt found" });
    const loan=debt.loanPurpose;
    if (!loan) return res.status(404).json({ message: "purpose not found" });

    const interest = (debt.remainingAmount * debt.interestRate) / 1200;
    const principal = amount - interest;

    debt.totalInterestPaid += interest;
    debt.remainingAmount -= principal;

    if (debt.remainingAmount <= 0) {
      debt.remainingAmount = 0;
      debt.status = "Paid";
    }
    await debt.save();



    const transaction = new Transaction({
      user: userId,
      type: "expense",
      category: "Debt Payment",
      amount,
      date: new Date(),
      description: `Paid ₹${amount} for debt ${loan} `,
    });

    await transaction.save();

    // ✅ Check if the user belongs to a family
    const familyTransaction = await FamilyTransaction.findOne({ _id:userfamily.familyId });

    if (familyTransaction) {
      familyTransaction.totalExpense += amount;
      familyTransaction.transactions.push({
        user: userId,
        username: user.username,
        type: "expense",
        category: "Debt Payment",
        amount,
        paymentMethod: "UPI",
        date: new Date(),
        description: `Paid ₹${amount} for family debt ${loan}`,
      });

    }else{
     familyTransaction = new FamilyTransaction({
            _id: userfamily.familyId,
            totalExpense: amount,
            transactions: [
              {
                user: userId,
                username:user.username, // TODO: Fetch user name from DB
                type: "expense",
                category: "Debt Paymen",
                amount,
                paymentMethod: "UPI",
                date: new Date(),
                description: `Paid ₹${amount} for family debt ${loan}`,
              },
            ],
          });
        }
        await familyTransaction.save();

    res.json(debt);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};



// ✅ Remove Debt
exports.deleteDebt = async (req, res) => {
  try {
    const { debtId } = req.params;

    // ✅ Validate ObjectId
    if (!mongoose.Types.ObjectId.isValid(debtId)) {
      return res.status(400).json({ message: "Invalid Debt ID" });
    }

    // ✅ Find and delete debt
    const debt = await Debt.findOneAndDelete({ _id: debtId, userId: req.user });

    if (!debt) return res.status(404).json({ message: "Debt not found" });

    res.json({ message: "Debt removed successfully" });
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};
