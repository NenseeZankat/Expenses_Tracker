const mongoose = require("mongoose");

const FamilySchema = new mongoose.Schema({
    familyName: String,
    admin: { type: mongoose.Schema.Types.ObjectId, ref: "User" }, // Admin of the family
    members: [{ type: mongoose.Schema.Types.ObjectId, ref: "User" }], // List of family members
    createdAt: { type: Date, default: Date.now }
});

const Family = mongoose.model("Family", FamilySchema);
module.exports = Family;
