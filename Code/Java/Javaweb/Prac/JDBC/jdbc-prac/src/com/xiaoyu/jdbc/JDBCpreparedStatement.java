package com.xiaoyu.jdbc;

import java.sql.*;

public class JDBCpreparedStatement {
    // Prevent SQL injection
    public static void main(String[] args) throws Exception {
        // Register the driver
        Class.forName("com.mysql.cj.jdbc.Driver");

        // Establish the connection
        String url = "jdbc:mysql://localhost:3306/prac?useServerPrepStmts=true"; // Use server-side prepared statements
        String user = "root";
        String password = "MYSQLc-137040808";
        Connection conn = DriverManager.getConnection(url, user, password);

        // Username and password
        String username = "larry";
        String pwd = "123456";

        // Define the SQL statement
        String sql = "select * from user_info where username = ? and password = ?";

        // Get the prepared statement
        PreparedStatement pstmt = conn.prepareStatement(sql);

        // Set ? values
        pstmt.setString(1, username);
        pstmt.setString(2, pwd);

        // Execute the SQL statement
        ResultSet rst = pstmt.executeQuery();

        if (rst.next()) {
            System.out.println("Login successful");
        } else {
            System.out.println("Login failed");
        }

        // Close the resources
        rst.close();
        pstmt.close();
        conn.close();
    }
}
