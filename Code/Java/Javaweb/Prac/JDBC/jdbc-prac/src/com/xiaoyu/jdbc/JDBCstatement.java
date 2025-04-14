package com.xiaoyu.jdbc;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.Statement;

public class JDBCstatement {
    public static void main(String[] args) throws Exception {
        // 1. Register the driver
        Class.forName("com.mysql.cj.jdbc.Driver");

        // 2. Establish the connection
        String url = "jdbc:mysql://localhost:3306/prac";
        String user = "root";
        String password = "MYSQLc-137040808";
        Connection conn = DriverManager.getConnection(url, user, password);

        // 3. Define the SQL statement
        String sql = "CREATE DATABASE db1";

        // 4. Create a statement
        Statement stmt = conn.createStatement();

        int rowsAffected = stmt.executeUpdate(sql);

        if (rowsAffected > 0) {
            System.out.println("Transaction committed successfully.");
        } else {
            System.out.println("Transaction failed.");
        }

        // 11. Close the resources
        stmt.close();
        conn.close();
    }
}
