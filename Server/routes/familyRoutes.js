const express = require("express");
const { createFamily, addFamilyMember, removeFamilyMember, assignRole, getFamilyDetails, getFamilyExpenses,getFamilyByUserId } = require("../controllers/familyController");
const isAuthenticated = require("../middlewares/isAuth");

const router = express.Router();

router.post("/add-member", addFamilyMember);      // Add membergetFamilyExpenses
router.post("/remove-member", removeFamilyMember); // Remove member
router.post("/assign-role", assignRole);          // Assign role
router.get("/:familyId", getFamilyDetails);       // Get family details
router.post("/create",  createFamily);             // Create family
router.get("/family/family-expenses",isAuthenticated, getFamilyExpenses);
router.get("/getFamilyByUserId/:userId", getFamilyByUserId);
module.exports = router;
