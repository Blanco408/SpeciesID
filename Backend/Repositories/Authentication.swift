//
//  Authentication.swift
//  
//
//  Created by William  Blanco  on 12/11/25.
//

import Foundation
import FirebaseAuth // library that handles logging in

actor Authentication {
    //registers a new user
    func register_new_user(email:String, password: String) async throws -> String{
        let result = try await Auth.auth().createUser(withEmail: email, password: password)
        return result.user.uid // uid is how a user is identified in firebase
    }
    
    func login(email:String, password: String) async throws -> String{
        let result = try await Auth.auth().signIn(withEmail: email, password: password)
        return result.user.uid
    }
    
    func current_user_uid() -> String? {
        return Auth.auth().currentUser?.uid
    }
    
    func logout() throws {
        try Auth.auth().signOut()
    }
}
