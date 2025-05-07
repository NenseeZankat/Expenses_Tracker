const express = require("express");
const usersController = require("../controllers/usersCtrl");
const isAuthenticated = require("../middlewares/isAuth");
const multer = require("multer"); // ✅ Import Multer
const path = require("path");
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, "uploads/"); // ✅ Save files in "uploads/" directory
  },
  filename: function (req, file, cb) {
    cb(null, Date.now() + "_" + file.originalname); // ✅ Unique filename
  },
});

// Multer Middleware
const upload = multer({ storage: storage });


const userRouter = express.Router();
//!Register
userRouter.post("/api/v1/users/register", usersController.register);
//! Login
userRouter.post("/api/v1/users/login", usersController.login);
//!Profile
userRouter.get(
  "/api/v1/users/profile",
  isAuthenticated,
  usersController.profile
);

userRouter.get(
  "/api/v1/users/by-email/:email",
  usersController.findUserByEmail
);
//!change password
userRouter.put(
  "/api/v1/users/change-password",
  isAuthenticated,
  usersController.changeUserPassword
);
//!update profile
userRouter.put(
  "/api/v1/users/update-profile",
  isAuthenticated,
  upload.single("profilePic"), // ✅ Multer middleware for image upload
  usersController.updateUserProfile
);

userRouter.put("/upload-profile-pic",
   isAuthenticated, 
  upload.single('profilePic'), 
  usersController.uploadProfilePic
);

module.exports = userRouter;
