const express = require("express");
const {
  createFamilyGoal,
  getFamilyGoals,
  joinFamilyGoal,
  updateContribution,
  deleteFamilyGoal,
} = require("../controllers/familyGoalController");

const router = express.Router();

// ✅ Create a new challenge
router.post("/", createFamilyGoal);

// ✅ Get all challenges for a family
router.get("/:familyId", getFamilyGoals);

// ✅ Join a challenge
router.put("/:goalId/join", joinFamilyGoal);

// ✅ Update contribution
router.put("/:goalId/contribute", updateContribution);

// ✅ Delete a challenge
router.delete("/:goalId", deleteFamilyGoal);

module.exports = router;
