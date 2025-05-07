
const Family = require("../model/Family");
const User = require("../model/User");
const Transaction = require("../model/Transaction"); // Ensure correct model import
const mongoose = require("mongoose");


// ✅ Create a new family
// exports.createFamily = async (req, res) => {
//     try {
//         const { familyName, adminId } = req.body;

//         if (!familyName || !adminId) {
//             return res.status(400).json({ message: "Family name and admin ID are required" });
//         }

//         const existingFamily = await Family.findOne({ familyName });
//         if (existingFamily) {
//             return res.status(400).json({ message: "Family with this name already exists" });
//         }

//         const newFamily = new Family({ familyName, admin: adminId, members: [adminId] });
//         await newFamily.save();

//         await User.findByIdAndUpdate(adminId, { familyId: newFamily._id, role: "admin" });

//         res.status(201).json({ message: "Family created", family: newFamily });
//     } catch (error) {
//         res.status(500).json({ message: "Error creating family", error });
//     }
// };

exports.createFamily = async (req, res) => {
    try {
        const { familyName, adminId } = req.body;

        if (!familyName || !adminId) {
            return res.status(400).json({ message: "Family name and admin ID are required" });
        }

        // ✅ Step 1: Check if the user is already part of a family
        const existingUser = await User.findById(adminId).select("familyId");
        if (existingUser && existingUser.familyId) {
            return res.status(400).json({ message: "User is already part of a family and cannot create another one" });
        }

        // ✅ Step 2: Check if the family name already exists
        const existingFamily = await Family.findOne({ familyName });
        if (existingFamily) {
            return res.status(400).json({ message: "Family with this name already exists" });
        }

        // ✅ Step 3: Create a new family
        const newFamily = new Family({ familyName, admin: adminId, members: [adminId] });
        await newFamily.save();

        // ✅ Step 4: Update the user with the new family ID and role
        await User.findByIdAndUpdate(adminId, { familyId: newFamily._id, role: "admin" });

        res.status(201).json({ message: "Family created successfully", family: newFamily });
    } catch (error) {
        res.status(500).json({ message: "Error creating family", error });
    }
};


// ✅ Add a member
exports.addFamilyMember = async (req, res) => {
    try {
        const { familyId, userId, adminId } = req.body;

        if (!familyId || !userId || !adminId) {
            return res.status(400).json({ message: "Family ID, user ID, and admin ID are required" });
        }

        const family = await Family.findById(familyId);
        if (!family || family.admin.toString() !== adminId) {
            return res.status(403).json({ message: "Only admin can add members" });
        }

        if (family.members.includes(userId)) return res.status(400).json({ message: "User already in family" });

        family.members.push(userId);
        await family.save();

        await User.findByIdAndUpdate(userId, { familyId });

        res.status(200).json({ message: "Member added", family });
    } catch (error) {
        res.status(500).json({ message: "Error adding member", error });
    }
};

// ✅ Remove a member
exports.removeFamilyMember = async (req, res) => {
    try {
        const { familyId, userId, adminId } = req.body;

        if (!familyId || !userId || !adminId) {
            return res.status(400).json({ message: "Family ID, user ID, and admin ID are required" });
        }

        const family = await Family.findById(familyId);
        if (!family || family.admin.toString() !== adminId) {
            return res.status(403).json({ message: "Only admin can remove members" });
        }

        family.members = family.members.filter(member => member.toString() !== userId);
        await family.save();

        await User.findByIdAndUpdate(userId, { familyId: null, role: "member" });

        res.status(200).json({ message: "Member removed", family });
    } catch (error) {
        res.status(500).json({ message: "Error removing member", error });
    }
};

// ✅ Assign role
exports.assignRole = async (req, res) => {
    try {
        const { familyId, userId, adminId, role } = req.body;

        if (!familyId || !userId || !adminId || !role) {
            return res.status(400).json({ message: "Family ID, user ID, admin ID, and role are required" });
        }

        if (!["admin", "member"].includes(role)) return res.status(400).json({ message: "Invalid role" });

        const family = await Family.findById(familyId);
        if (!family || family.admin.toString() !== adminId) {
            return res.status(403).json({ message: "Only admin can assign roles" });
        }

        await User.findByIdAndUpdate(userId, { role });

        res.status(200).json({ message: "Role updated" });
    } catch (error) {
        res.status(500).json({ message: "Error updating role", error });
    }
};

// ✅ Get family details
exports.getFamilyDetails = async (req, res) => {
    try {
        const { familyId } = req.params;

        if (!familyId) {
            return res.status(400).json({ message: "Family ID is required" });
        }

        const family = await Family.findById(familyId).populate("members", "username email role");
        if (!family) return res.status(404).json({ message: "Family not found" });

        res.status(200).json({ family });
    } catch (error) {
        res.status(500).json({ message: "Error fetching family", error });
    }
};

// ✅ Get family expenses
exports.getFamilyExpenses = async (req, res) => {
    try {
        if (!req.user) {
            return res.status(401).json({ message: "User not authenticated" });
        }

        const user = await User.findById(req.user);
        if (!user || !user.familyId) {
            return res.status(400).json({ message: "User is not part of a family" });
        }

        const family = await Family.findById(user.familyId).populate("members", "_id");
        if (!family) {
            return res.status(404).json({ message: "Family not found" });
        }

        const memberIds = family.members.map(member => member._id);

        const transactions = await Transaction.aggregate([
            { $match: { user: { $in: memberIds }, type: "expense" } },
            {
                $group: {
                    _id: "$category",
                    totalAmount: { $sum: "$amount" },
                },
            },
        ]);

        res.json(transactions);
    } catch (error) {
        res.status(500).json({ message: "Error fetching family expenses", error });
    }
};

// ✅ Get family members by user ID
exports.getFamilyByUserId = async (req, res) => {
    try {
        const { userId } = req.params;

        if (!userId) {
            return res.status(400).json({ message: "User ID is required" });
        }

        if (!mongoose.Types.ObjectId.isValid(userId)) {
            return res.status(400).json({ message: "Invalid user ID format" });
        }

        // const user = await User.findById(userId).select("familyId");
        // if (!user || !user.familyId) {
        //     return res.status(400).json({ message: "User is not part of any family" });
        // }

        const user = await User.findById(userId).select("familyId");
        console.log("User Family ID:", user?.familyId);  // Debugging log


        const family = await Family.findById(user.familyId).populate("members", "username email");
        if (!family) return res.status(404).json({ message: "Family not found" });

        res.status(200).json({ members: family.members });
    } catch (error) {
        // res.status(500).json({ message: "Error fetching family members", error });
        console.error("Error fetching family members:", error); // Logs the error in the backend
        res.status(500).json({ message: "Error fetching family members", error: error.message });
    }
};
      

