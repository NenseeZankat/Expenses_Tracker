const FamilyTransaction = require("../model/FamilyTransaction");

// Get all family transactions
exports.getFamilyTransactions = async (req, res) => {
  try {
    const familyTransactions = await FamilyTransaction.findById(req.params.familyId);
    if (!familyTransactions) return res.status(404).json({ message: "Family transactions not found" });

    res.json(familyTransactions);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};
