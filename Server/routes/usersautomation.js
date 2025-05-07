const express = require("express");
const router = express.Router();
const User = require("../model/User");

// Get user allocation preference
router.get("/api/v1/:id", async (req, res) => {
    try {
        const user = await User.findById(req.params.id);
        if (!user) return res.status(404).json({ message: "User not found" });

        res.json({ allocationPreference: user.allocationPreference });
    } catch (error) {
        res.status(500).json({ message: error.message });
    }
});

// Update allocation preference
router.put("/api/v1/:id/allocation-preference", async (req, res) => {
    try {
        const { allocationPreference } = req.body;
        await User.findByIdAndUpdate(req.params.id, { allocationPreference });

        res.json({ message: "Preference updated successfully" });
    } catch (error) {
        res.status(500).json({ message: error.message });
    }
});

module.exports = router;
