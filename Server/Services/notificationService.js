const nodemailer = require("nodemailer");
const User = require("../model/User"); // Assuming a user model exists

async function sendNotification(userId, message) {
  const user = await User.findById(userId);
  if (!user) return;

  let transporter = nodemailer.createTransport({
    service: "gmail",
    auth: { user: "srushti0906@gmail.com", pass: "dlwwplkoqjcqiqhb" },
  });

  await transporter.sendMail({
    from: '"Subscription Alert" <your-email@gmail.com>',
    to: user.email,
    subject: "Subscription Renewal Alert",
    text: message,
  });

  console.log("Notification sent to:", user.email);
}

module.exports = sendNotification;
