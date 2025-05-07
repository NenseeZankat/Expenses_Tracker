const mongoose = require("mongoose");

const userSchema = new mongoose.Schema(
  {
    username: {
      type: String,
      required: true,
      unique: true,
    },
    email: {
      type: String,
      required: true,
      unique: true,
    },
    password: {
      type: String,
      required: true,
    }, 
    role:
     { type: String, enum: ["admin", "member","child","individual"]}, 
     // Role in family
    familyId: { type: mongoose.Schema.Types.ObjectId, ref: "Family", default: null }, 
    // Family reference
    createdAt: { type: Date, default: Date.now },
    
    allocationPreference: {
      type: String,
      enum: ["next_month", "family_goal"],
      default: "next_month", // Default preference is rolling over to next month
    },

    profilePic: { type: String, default: "" } // Store image URL or filename

  },
  
  {
    timestamps: true,
  }
);

module.exports = mongoose.model("User", userSchema);
